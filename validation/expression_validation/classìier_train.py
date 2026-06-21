from pathlib import Path
import sys
import re
import random
import json

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


DATA_ROOT = ROOT / "datasets" / "processed_KDEF"
OUT_DIR = ROOT / "validation" / "expression_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = OUT_DIR / "expression_classifier_best.pth"
LOG_CSV = OUT_DIR / "classifier_train_log.csv"

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
SEED = 42
TRAIN_RATIO = 0.8

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_identity(path: Path):
    m = re.match(r"^(\d+)_", path.name)
    if m is None:
        return None
    return int(m.group(1))


def collect_samples():
    rows = []

    for label in CLASSES:
        cls_dir = DATA_ROOT / label

        for path in sorted(cls_dir.glob("*_2.*")):
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            identity = parse_identity(path)

            if identity is None:
                continue

            rows.append({
                "path": str(path),
                "label": label,
                "label_idx": CLASS_TO_IDX[label],
                "identity": identity,
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(f"Không tìm thấy ảnh frontal *_2 trong {DATA_ROOT}")

    return df


def split_by_identity(df):
    ids = sorted(df["identity"].unique().tolist())
    random.shuffle(ids)

    n_train = int(len(ids) * TRAIN_RATIO)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:])

    train_df = df[df["identity"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["identity"].isin(val_ids)].reset_index(drop=True)

    return train_df, val_df


class ExpressionDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)

        if train:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["path"]).convert("RGB")
        img = self.tf(img)

        y = int(row["label_idx"])

        return img, torch.tensor(y, dtype=torch.long)


def build_model():
    try:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        print("Loaded ResNet18 ImageNet pretrained.")
    except Exception:
        model = resnet18(weights=None)
        print("Warning: dùng ResNet18 weights=None.")

    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()

    f1s = []

    for c in range(len(CLASSES)):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))

    return float(acc), macro_f1


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        pred = logits.argmax(dim=1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc, macro_f1 = compute_metrics(y_true, y_pred)

    return avg_loss, acc, macro_f1


def train():
    set_seed(SEED)

    df = collect_samples()

    print("Total images:", len(df))
    print(df["label"].value_counts())

    train_df, val_df = split_by_identity(df)

    print("\nTrain images:", len(train_df))
    print("Val images:", len(val_df))
    print("Train identities:", train_df["identity"].nunique())
    print("Val identities:", val_df["identity"].nunique())

    train_set = ExpressionDataset(train_df, train=True)
    val_set = ExpressionDataset(val_df, train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    logs = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        y_true = []
        y_pred = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for x, y in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

            pred = logits.argmax(dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

            train_acc, train_f1 = compute_metrics(y_true, y_pred)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{train_acc:.4f}",
                "f1": f"{train_f1:.4f}",
            })

        train_loss = total_loss / len(train_loader.dataset)
        train_acc, train_f1 = compute_metrics(y_true, y_pred)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })

        pd.DataFrame(logs).to_csv(LOG_CSV, index=False)

        if val_f1 > best_f1:
            best_f1 = val_f1

            torch.save({
                "model": model.state_dict(),
                "classes": CLASSES,
                "img_size": IMG_SIZE,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "epoch": epoch,
            }, CKPT_PATH)

            print("Saved best:", CKPT_PATH)

    with open(OUT_DIR / "classes.json", "w") as f:
        json.dump(CLASSES, f, indent=2)

    print("\nDone.")
    print("Best val macro F1:", best_f1)
    print("Checkpoint:", CKPT_PATH)
    print("Log:", LOG_CSV)


if __name__ == "__main__":
    train()