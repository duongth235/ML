from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from config import *
from model import GeometryAutoEncoder, Generator


DEVICE = torch.device(DEVICE)

VAL_DIR = Path("validation")
GEN_DIR = VAL_DIR / "gen_data"
GEN_DIR.mkdir(parents=True, exist_ok=True)

NUM_PER_EXPR = 30
G_CKPT = OUT_DIR / "G_140.pth"
E_CKPT = OUT_DIR / "E.pth"
LANDMARK_CSV = OUT_DIR / "landmarks_front_pose.csv"


def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2


tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def load_img(path):
    return tf(Image.open(path).convert("RGB"))


def get_g(row):
    arr = row[[f"g{i}" for i in range(136)]].values.astype("float32")
    return torch.tensor(arr, dtype=torch.float32)


def pick_source_target(df, expr):
    target_pool = df[df["label"] == expr]

    if len(target_pool) == 0:
        raise RuntimeError(f"Không có target expression: {expr}")

    while True:
        src_row = df.sample(1).iloc[0]
        tgt_row = target_pool.sample(1).iloc[0]

        if int(src_row["identity"]) != int(tgt_row["identity"]):
            return src_row, tgt_row


@torch.no_grad()
def main():
    df = pd.read_csv(LANDMARK_CSV)

    expressions = sorted(df["label"].unique().tolist())

    E = GeometryAutoEncoder(input_dim=136, z_g_dim=Z_G_DIM).to(DEVICE)
    G = Generator(z_i_dim=Z_I_DIM, z_g_dim=Z_G_DIM).to(DEVICE)

    E.load_state_dict(torch.load(E_CKPT, map_location=DEVICE))
    G.load_state_dict(torch.load(G_CKPT, map_location=DEVICE))

    E.eval()
    G.eval()

    used_names = set()

    for expr in expressions:
        expr_out_dir = GEN_DIR / expr
        expr_out_dir.mkdir(parents=True, exist_ok=True)

        count = 0

        while count < NUM_PER_EXPR:
            src_row, tgt_row = pick_source_target(df, expr)

            src_id = int(src_row["identity"])
            tgt_id = int(tgt_row["identity"])

            src_img = load_img(src_row["path"]).unsqueeze(0).to(DEVICE)
            tgt_g = get_g(tgt_row).unsqueeze(0).to(DEVICE)

            z_g, _ = E(tgt_g)
            fake = G(src_img, z_g)

            # tên: sourceID_expression_index.png
            out_name = f"{src_id}_{expr}_{count:03d}.png"

            # tránh trùng nếu random lại cùng source id
            if out_name in used_names:
                continue

            used_names.add(out_name)
            save_image(denorm(fake), expr_out_dir / out_name)

            print(
                f"[{expr}] saved {out_name} | "
                f"source_id={src_id}, target_id={tgt_id}"
            )

            count += 1

    print(f"\nDone. Saved to: {GEN_DIR}")


if __name__ == "__main__":
    main()