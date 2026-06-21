from pathlib import Path
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

import cv2
from insightface.app import FaceAnalysis


DATA_ROOT = Path("datasets/processed_KDEF")
OUT_FILE = Path("validation/id_validation/gallery_embeddings.npz")
OUT_CSV = Path("validation/id_validation/gallery_build_report.csv")


def parse_identity(filename):
    m = re.match(r"^(\d+)_", filename)
    return int(m.group(1)) if m else None


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def center_square_resize(rgb, size=112):
    h, w = rgb.shape[:2]
    side = min(h, w)

    y1 = (h - side) // 2
    x1 = (w - side) // 2

    crop = rgb[y1:y1 + side, x1:x1 + side]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    return crop


def get_embedding_direct(app, path):
    """
    Không detect mặt.
    Giả định ảnh đã crop/align mặt tương đối tốt.
    Đưa thẳng ảnh 112x112 vào recognition model.
    """
    rgb = load_rgb(path)
    rgb = center_square_resize(rgb, 112)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    rec_model = app.models["recognition"]

    emb = rec_model.get_feat(bgr).flatten().astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    return emb


def main():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    app.prepare(ctx_id=-1, det_size=(640, 640))

    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
        for path in DATA_ROOT.rglob(ext):
            if path.stem.endswith("_2"):
                image_paths.append(path)

    image_paths = sorted(image_paths)

    print("num frontal images =", len(image_paths))

    by_identity = {}
    rows = []

    for path in tqdm(image_paths, desc="Build gallery direct"):
        identity = parse_identity(path.name)

        if identity is None:
            rows.append({
                "path": str(path),
                "filename": path.name,
                "identity": None,
                "success": False,
                "reason": "cannot_parse_identity",
            })
            continue

        try:
            emb = get_embedding_direct(app, path)

            by_identity.setdefault(identity, []).append(emb)

            rows.append({
                "path": str(path),
                "filename": path.name,
                "identity": identity,
                "success": True,
                "reason": "ok",
            })

        except Exception as e:
            rows.append({
                "path": str(path),
                "filename": path.name,
                "identity": identity,
                "success": False,
                "reason": str(e),
            })

    gallery = {}

    for identity, embs in by_identity.items():
        mean_emb = np.mean(embs, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        gallery[str(identity)] = mean_emb.astype(np.float32)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(OUT_FILE, **gallery)

    import pandas as pd
    report = pd.DataFrame(rows)
    report.to_csv(OUT_CSV, index=False)

    total = len(report)
    success = int(report["success"].sum())
    failed = total - success

    print()
    print("========== Gallery Build Direct ==========")
    print("total images        :", total)
    print("success images      :", success)
    print("failed images       :", failed)
    print("success rate        :", f"{success / total:.4f}" if total > 0 else "0.0000")
    print("gallery identities  :", len(gallery))
    print("saved npz           :", OUT_FILE)
    print("saved report        :", OUT_CSV)

    print()
    print("identity image counts:")


if __name__ == "__main__":
    main()