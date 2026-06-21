from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import cv2
from insightface.app import FaceAnalysis


GEN_DIR = Path("validation/gen_data")
GALLERY_NPZ = Path("validation/id_validation/gallery_embeddings.npz")

OUT_CSV = Path("validation/id_validation/id_eval_results.csv")
OUT_SUMMARY = Path("validation/id_validation/id_eval_summary.txt")


def parse_source_id(filename):
    m = re.match(r"^(\d+)_", Path(filename).name)
    if m is None:
        return None
    return int(m.group(1))


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
    rgb = load_rgb(path)
    rgb = center_square_resize(rgb, 112)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    rec_model = app.models["recognition"]

    emb = rec_model.get_feat(bgr).flatten().astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    return emb


def main():
    gallery_npz = np.load(GALLERY_NPZ)

    gallery_ids = sorted([int(k) for k in gallery_npz.files])
    gallery_mat = np.stack(
        [gallery_npz[str(i)] for i in gallery_ids],
        axis=0
    ).astype(np.float32)

    gallery_mat = gallery_mat / (
        np.linalg.norm(gallery_mat, axis=1, keepdims=True) + 1e-8
    )

    print("gallery identities:", len(gallery_ids))
    print("gallery matrix:", gallery_mat.shape)

    fake_paths = sorted(
        list(GEN_DIR.rglob("*.png")) +
        list(GEN_DIR.rglob("*.jpg")) +
        list(GEN_DIR.rglob("*.jpeg"))
    )

    print("num generated images:", len(fake_paths))

    if len(fake_paths) == 0:
        raise RuntimeError(f"Không tìm thấy ảnh trong {GEN_DIR}")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    rows = []

    for path in tqdm(fake_paths, desc="Evaluate ID"):
        source_id = parse_source_id(path.name)

        if source_id is None:
            rows.append({
                "path": str(path),
                "filename": path.name,
                "source_id": None,
                "source_in_gallery": False,
                "valid_eval": False,
                "id_cosine": np.nan,
                "rank1_id": None,
                "rank1_cosine": np.nan,
                "rank1_correct": False,
                "rank3_correct": False,
                "reason": "cannot_parse_source_id",
            })
            continue

        source_in_gallery = source_id in gallery_ids

        try:
            fake_emb = get_embedding_direct(app, path)
        except Exception as e:
            rows.append({
                "path": str(path),
                "filename": path.name,
                "source_id": source_id,
                "source_in_gallery": source_in_gallery,
                "valid_eval": False,
                "id_cosine": np.nan,
                "rank1_id": None,
                "rank1_cosine": np.nan,
                "rank1_correct": False,
                "rank3_correct": False,
                "reason": str(e),
            })
            continue

        sims = gallery_mat @ fake_emb

        order = np.argsort(-sims)
        rank1_idx = int(order[0])
        rank1_id = int(gallery_ids[rank1_idx])
        rank1_cosine = float(sims[rank1_idx])

        top3_ids = [
            int(gallery_ids[int(i)])
            for i in order[:3]
        ]

        if source_in_gallery:
            source_idx = gallery_ids.index(source_id)
            id_cosine = float(sims[source_idx])
            rank1_correct = rank1_id == source_id
            rank3_correct = source_id in top3_ids
            valid_eval = True
            reason = "ok"
        else:
            id_cosine = np.nan
            rank1_correct = False
            rank3_correct = False
            valid_eval = False
            reason = "source_not_in_gallery"

        rows.append({
            "path": str(path),
            "filename": path.name,
            "source_id": source_id,
            "source_in_gallery": source_in_gallery,
            "valid_eval": valid_eval,
            "id_cosine": id_cosine,
            "rank1_id": rank1_id,
            "rank1_cosine": rank1_cosine,
            "rank1_correct": rank1_correct,
            "rank3_correct": rank3_correct,
            "top3_ids": str(top3_ids),
            "reason": reason,
        })

    result = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    valid = result[result["valid_eval"] == True]

    num_images = len(result)
    num_valid = len(valid)

    valid_rate = num_valid / num_images if num_images > 0 else 0.0

    mean_cos = valid["id_cosine"].mean()
    rank1 = valid["rank1_correct"].mean()
    rank3 = valid["rank3_correct"].mean()

    summary = f"""
========== ID Evaluation ==========
num_images              : {num_images}
num_valid_eval           : {num_valid}
valid_rate               : {valid_rate:.4f}

mean_id_cosine           : {mean_cos:.4f}
rank1_accuracy           : {rank1:.4f}
rank3_accuracy           : {rank3:.4f}

gallery identities       : {len(gallery_ids)}
gallery file             : {GALLERY_NPZ}

saved csv                : {OUT_CSV}
"""

    print(summary)

    with open(OUT_SUMMARY, "w") as f:
        f.write(summary)

    print("saved summary:", OUT_SUMMARY)


if __name__ == "__main__":
    main()