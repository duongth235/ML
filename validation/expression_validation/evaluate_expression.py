from pathlib import Path
import sys
import re
import json

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18


GEN_DIR = ROOT / "validation" / "gen_data"
CKPT_PATH = ROOT / "validation" / "expression_validation" / "expression_classifier_best.pth"

OUT_DIR = ROOT / "validation" / "expression_validation"
OUT_CSV = OUT_DIR / "expression_eval_results.csv"
OUT_SUMMARY = OUT_DIR / "expression_eval_summary.txt"
CONFUSION_CSV = OUT_DIR / "expression_confusion_matrix.csv"
PER_CLASS_CSV = OUT_DIR / "expression_per_class_accuracy.csv"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

DEFAULT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def parse_expr_from_filename(filename):
    """
    Filename dạng:
        12_happy_003.png
    => expression = happy
    """
    name = Path(filename).name
    m = re.match(r"^\d+_([A-Za-z]+)_\d+\.(png|jpg|jpeg)$", name)

    if m is None:
        return None

    return m.group(1)


def load_classes(ckpt):
    if "classes" in ckpt:
        return ckpt["classes"]

    classes_json = OUT_DIR / "classes.json"

    if classes_json.exists():
        with open(classes_json, "r") as f:
            return json.load(f)

    return DEFAULT_CLASSES


def build_model(num_classes):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_macro_f1(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1s.append(f1)

    return float(np.mean(f1s)), f1s


def build_confusion(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for gt, pred in zip(y_true, y_pred):
        cm[gt, pred] += 1

    return cm


@torch.no_grad()
def main():
    if not CKPT_PATH.exists():
        raise RuntimeError(f"Không thấy checkpoint: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    classes = load_classes(ckpt)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    img_size = ckpt.get("img_size", 224)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model = build_model(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    fake_paths = sorted(
        list(GEN_DIR.rglob("*.png")) +
        list(GEN_DIR.rglob("*.jpg")) +
        list(GEN_DIR.rglob("*.jpeg"))
    )

    if len(fake_paths) == 0:
        raise RuntimeError(f"Không tìm thấy ảnh gen trong {GEN_DIR}")

    rows = []
    y_true = []
    y_pred = []

    for path in tqdm(fake_paths, desc="Evaluate expression"):
        gt_expr = parse_expr_from_filename(path.name)

        if gt_expr is None or gt_expr not in class_to_idx:
            rows.append({
                "path": str(path),
                "filename": path.name,
                "gt_expr": gt_expr,
                "pred_expr": None,
                "correct": False,
                "confidence": np.nan,
                "reason": "invalid_filename_or_class",
            })
            continue

        img = Image.open(path).convert("RGB")
        x = tf(img).unsqueeze(0).to(DEVICE)

        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]

        pred_idx = int(prob.argmax().item())
        gt_idx = int(class_to_idx[gt_expr])

        pred_expr = classes[pred_idx]
        confidence = float(prob[pred_idx].item())
        correct = pred_idx == gt_idx

        y_true.append(gt_idx)
        y_pred.append(pred_idx)

        rows.append({
            "path": str(path),
            "filename": path.name,
            "gt_expr": gt_expr,
            "pred_expr": pred_expr,
            "correct": correct,
            "confidence": confidence,
            "reason": "ok",
        })

    result = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    if len(y_true) == 0:
        raise RuntimeError("Không có ảnh hợp lệ để đánh giá expression.")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float((y_true == y_pred).mean())
    macro_f1, f1s = compute_macro_f1(y_true, y_pred, len(classes))

    cm = build_confusion(y_true, y_pred, len(classes))

    cm_df = pd.DataFrame(
        cm,
        index=[f"gt_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )
    cm_df.to_csv(CONFUSION_CSV)

    per_class_rows = []

    for i, cls in enumerate(classes):
        total = int((y_true == i).sum())
        correct = int(((y_true == i) & (y_pred == i)).sum())
        acc_i = correct / total if total > 0 else 0.0

        per_class_rows.append({
            "class": cls,
            "total": total,
            "correct": correct,
            "accuracy": acc_i,
            "f1": f1s[i],
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(PER_CLASS_CSV, index=False)

    summary = f"""
========== Expression Evaluation ==========
num_images              : {len(result)}
num_valid_eval           : {len(y_true)}

expression_accuracy      : {acc:.4f}
expression_macro_f1      : {macro_f1:.4f}

classes                  : {classes}

saved csv                : {OUT_CSV}
confusion matrix          : {CONFUSION_CSV}
per-class metrics         : {PER_CLASS_CSV}
"""

    print(summary)

    with open(OUT_SUMMARY, "w") as f:
        f.write(summary)

    print("saved summary:", OUT_SUMMARY)


if __name__ == "__main__":
    main()