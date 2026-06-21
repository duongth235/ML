import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from config import *
from model import GeometryAutoEncoder, Generator


DEVICE = torch.device(DEVICE)


def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2


tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def load_img(path):
    img = Image.open(path).convert("RGB")
    return tf(img)


def get_g(row):
    arr = row[[f"g{i}" for i in range(136)]].values.astype("float32")
    return torch.tensor(arr, dtype=torch.float32)


def tensor_to_img(x):
    x = denorm(x.detach().cpu())
    x = x.permute(1, 2, 0).numpy()
    return x


def random_pair(df):
    src = df.sample(1).iloc[0]
    tgt = df.sample(1).iloc[0]
    return src, tgt


def pick_by_path(df, path):
    path = str(Path(path))
    hit = df[df["path"] == path]

    if len(hit) == 0:
        hit = df[df["path"].apply(lambda p: Path(p).name == Path(path).name)]

    if len(hit) == 0:
        raise ValueError(f"Không tìm thấy ảnh trong landmark csv: {path}")

    return hit.iloc[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(OUT_DIR / "generated_result.jpg"))

    args = parser.parse_args()

    df = pd.read_csv(OUT_DIR / "landmarks_front_pose.csv")

    if args.source is None and args.target is None:
        src_row, tgt_row = random_pair(df)
    elif args.source is not None and args.target is not None:
        src_row = pick_by_path(df, args.source)
        tgt_row = pick_by_path(df, args.target)
    else:
        raise ValueError("Phải truyền cả --source và --target, hoặc không truyền gì để random.")

    E = GeometryAutoEncoder(input_dim=136, z_g_dim=Z_G_DIM).to(DEVICE)
    G = Generator(z_i_dim=Z_I_DIM, z_g_dim=Z_G_DIM).to(DEVICE)

    E.load_state_dict(torch.load(OUT_DIR / "E.pth", map_location=DEVICE))
    G.load_state_dict(torch.load(OUT_DIR / "G_140.pth", map_location=DEVICE))

    E.eval()
    G.eval()

    src_img = load_img(src_row["path"]).unsqueeze(0).to(DEVICE)
    tgt_img = load_img(tgt_row["path"]).unsqueeze(0).to(DEVICE)
    tgt_g = get_g(tgt_row).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        z_g, _ = E(tgt_g)
        fake = G(src_img, z_g)

    result = torch.cat([src_img, tgt_img, fake], dim=0)
    save_path = Path(args.out)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(
        denorm(result),
        save_path,
        nrow=3,
    )

    print("SOURCE:", src_row["label"], "id:", src_row["identity"])
    print("TARGET:", tgt_row["label"], "id:", tgt_row["identity"])
    print("saved:", save_path)

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_img(src_img[0]))
    plt.title(f"Source\n{src_row['label']} | id {src_row['identity']}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(tensor_to_img(tgt_img[0]))
    plt.title(f"Target\n{tgt_row['label']} | id {tgt_row['identity']}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(tensor_to_img(fake[0]))
    plt.title("Generated")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()