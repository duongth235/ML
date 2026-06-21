import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import face_alignment
from tqdm.auto import tqdm

from config import *
from model import GeometryAutoEncoder, Generator, Discriminator


OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device(DEVICE)


def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2


def detect_landmarks_cache():
    cache_path = OUT_DIR / "landmarks_front_pose.csv"

    if cache_path.exists():
        print("load landmark cache:", cache_path)
        return pd.read_csv(cache_path)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cpu",
        flip_input=False,
    )

    rows = []
    failed = []

    class_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])

    for cls_dir in tqdm(class_dirs, desc="Detect landmarks by class"):
        label = cls_dir.name

        for img_path in tqdm(sorted(cls_dir.glob("*_2.jpg")), desc=label, leave=False):
            try:
                identity = int(img_path.stem.split("_")[0])

                img = np.array(Image.open(img_path).convert("RGB"))
                preds = fa.get_landmarks(img)

                if preds is None:
                    failed.append(str(img_path))
                    continue

                lm = preds[0].astype(np.float32)

                h, w = img.shape[:2]
                lm[:, 0] = 2.0 * lm[:, 0] / w - 1.0
                lm[:, 1] = 2.0 * lm[:, 1] / h - 1.0

                row = {
                    "path": str(img_path),
                    "label": label,
                    "identity": identity,
                }

                for i, v in enumerate(lm.reshape(-1)):
                    row[f"g{i}"] = float(v)

                rows.append(row)

            except Exception as e:
                print("error:", img_path, e)
                failed.append(str(img_path))

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)

    failed_path = OUT_DIR / "detect_failed.txt"
    with open(failed_path, "w") as f:
        for p in failed:
            f.write(p + "\n")

    print("saved landmark cache:", cache_path)
    print("failed:", len(failed), "->", failed_path)

    return df


class KDEFTripletDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        self.tf = transforms.Compose([

                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02
                ),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        self.by_identity = {}
        self.by_label = {}

        for idx, row in self.df.iterrows():
            identity = int(row["identity"])
            label = str(row["label"])

            self.by_identity.setdefault(identity, []).append(idx)
            self.by_label.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.df)

    def load_img(self, path):
        return self.tf(Image.open(path).convert("RGB"))

    def get_g(self, row):
        arr = row[[f"g{i}" for i in range(136)]].values.astype(np.float32)
        return torch.tensor(arr, dtype=torch.float32)

    def sample_target_same_identity_diff_expr(self, src_idx):
        src_row = self.df.iloc[src_idx]
        identity = int(src_row["identity"])
        src_label = str(src_row["label"])

        candidates = [
            i for i in self.by_identity[identity]
            if str(self.df.iloc[i]["label"]) != src_label
        ]

        if len(candidates) == 0:
            return src_idx

        return random.choice(candidates)

    def sample_ref_random(self):
        return random.randrange(len(self.df))

    def __getitem__(self, idx):
        src_idx = idx
        tgt_idx = self.sample_target_same_identity_diff_expr(src_idx)
        ref_idx = self.sample_ref_random()

        src_row = self.df.iloc[src_idx]
        tgt_row = self.df.iloc[tgt_idx]
        ref_row = self.df.iloc[ref_idx]

        src_img = self.load_img(src_row["path"])
        tgt_img = self.load_img(tgt_row["path"])

        tgt_g = self.get_g(tgt_row)
        ref_g = self.get_g(ref_row)

        same_expr = float(str(tgt_row["label"]) == str(ref_row["label"]))
        src_g = self.get_g(src_row)

        return {
            "src_img": src_img,
            "tgt_img": tgt_img,

            "tgt_g": tgt_g,
            "ref_g": ref_g,

            "same_expr": torch.tensor(same_expr, dtype=torch.float32),

            "src_label": str(src_row["label"]),
            "tgt_label": str(tgt_row["label"]),
            "src_id": int(src_row["identity"]),
            "tgt_id": int(tgt_row["identity"]),
            "src_g": src_g
        }


def contrastive_loss(z1, z2, same, margin=MARGIN):
    dist = torch.sum((z1 - z2) ** 2, dim=1)

    pos = same * dist
    neg = (1.0 - same) * torch.clamp(margin - dist, min=0.0)

    return 0.5 * (pos + neg).mean()


def gradient_penalty(D, real, fake):
    bsz = real.size(0)

    eps = torch.rand(bsz, 1, 1, 1, device=real.device)
    mixed = eps * real + (1.0 - eps) * fake
    mixed.requires_grad_(True)

    score = D(mixed).mean(dim=1)

    grad = torch.autograd.grad(
        outputs=score,
        inputs=mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad = grad.view(bsz, -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def save_samples(E, G, loader, epoch):
    E.eval()
    G.eval()

    batch = next(iter(loader))

    src = batch["src_img"].to(DEVICE)
    tgt = batch["tgt_img"].to(DEVICE)
    tgt_g = batch["tgt_g"].to(DEVICE)

    with torch.no_grad():
        z_g, _ = E(tgt_g)
        fake = G(src, z_g)

    n = min(8, src.size(0))
    grid = torch.cat([src[:n], tgt[:n], fake[:n]], dim=0)

    save_path = OUT_DIR / f"sample_epoch_{epoch:03d}.jpg"
    save_image(denorm(grid), save_path, nrow=n)

    print("saved sample:", save_path)


def train_E(E, loader):
    print("\n========== Pretrain Geometry AutoEncoder E ==========")
    e_path = OUT_DIR / "E.pth"

    if e_path.exists():
        print(f"found pretrained E: {e_path}")
        E.load_state_dict(torch.load(e_path, map_location=DEVICE))
        E.eval()
        return
    opt = torch.optim.Adam(E.parameters(), lr=LR, betas=(BETA1, BETA2))

    for epoch in range(1, EPOCHS_E + 1):
        E.train()

        total_loss = 0.0
        total_contr = 0.0
        total_gr = 0.0

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"[E] Epoch {epoch}/{EPOCHS_E}")

        for step, batch in pbar:
            tgt_g = batch["tgt_g"].to(DEVICE)
            ref_g = batch["ref_g"].to(DEVICE)
            same = batch["same_expr"].to(DEVICE)

            z_tgt, rec_tgt = E(tgt_g)
            z_ref, rec_ref = E(ref_g)

            loss_contr = contrastive_loss(z_tgt, z_ref, same)
            loss_gr = F.mse_loss(rec_tgt, tgt_g) + F.mse_loss(rec_ref, ref_g)

            loss = LAMBDA_CONTR * loss_contr + LAMBDA_GR * loss_gr

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_contr += loss_contr.item()
            total_gr += loss_gr.item()

            pbar.set_postfix({
                "loss": f"{total_loss / (step + 1):.4f}",
                "contr": f"{total_contr / (step + 1):.4f}",
                "gr": f"{total_gr / (step + 1):.4f}",
            })

        torch.save(E.state_dict(), OUT_DIR / "E.pth")

    print("saved:", OUT_DIR / "E.pth")


def instance_norm_image(x, eps=1e-6):
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + eps)


def train_GAN(E, G, D, loader):
    print("\n========== Train G + D ==========")

    E.eval()
    for p in E.parameters():
        p.requires_grad_(False)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    for epoch in range(1, EPOCHS_GAN + 1):
        G.train()
        D.train()

        total_g = 0.0
        total_d = 0.0
        total_ir = 0.0
        total_adv = 0.0
        total_gp = 0.0

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"[GAN] Epoch {epoch}/{EPOCHS_GAN}")

        for step, batch in pbar:
            src = batch["src_img"].to(DEVICE)
            tgt = batch["tgt_img"].to(DEVICE)
            tgt_g = batch["tgt_g"].to(DEVICE)

            with torch.no_grad():
                z_g, _ = E(tgt_g)

            # train D
            fake = G(src, z_g).detach()

            d_real = D(tgt).mean()
            d_fake = D(fake).mean()
            gp = gradient_penalty(D, tgt, fake)

            loss_D = d_fake - d_real + 10.0 * gp

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # train G
            fake = G(src, z_g)

            loss_gen = -D(fake).mean()

            loss_rgb = F.l1_loss(fake, tgt)

            fake_n = instance_norm_image(fake)
            tgt_n = instance_norm_image(tgt)
            loss_norm = F.l1_loss(fake_n, tgt_n)

            loss_ir = LAMBDA_RGB * loss_rgb + LAMBDA_NORM * loss_norm

            loss_color = F.l1_loss(
                fake.mean(dim=[2, 3]),
                src.mean(dim=[2, 3])
            )

            loss_G = LAMBDA_IR * loss_ir + LAMBDA_ADV * loss_gen + LAMBDA_COLOR * loss_color

            with torch.no_grad():
                src_g, _ = E(batch["src_g"].to(DEVICE))

            fake_self = G(src, src_g)
            loss_self = F.l1_loss(fake_self, src)

            loss_G = loss_G + LAMBDA_ID * loss_self 

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            total_g += loss_G.item()
            total_d += loss_D.item()
            total_ir += loss_ir.item()
            total_adv += loss_gen.item()
            total_gp += gp.item()

            pbar.set_postfix({
                "G": f"{total_g / (step + 1):.4f}",
                "D": f"{total_d / (step + 1):.4f}",
                "IR": f"{total_ir / (step + 1):.4f}",
                "ADV": f"{total_adv / (step + 1):.4f}",
                "GP": f"{total_gp / (step + 1):.4f}",
            })

        if (epoch == 1 or epoch % 5 == 0):
            save_samples(E, G, loader, epoch)
            if epoch >= 2/3 * EPOCHS_GAN:
                torch.save(
                    G.state_dict(),
                    OUT_DIR / f"G_{epoch:03d}.pth"
                )

                torch.save(
                    D.state_dict(),
                    OUT_DIR / f"D_{epoch:03d}.pth"
                )


def main():
    df = detect_landmarks_cache()

    print("\n========== Dataset ==========")
    print("num front-view images:", len(df))
    print(df["label"].value_counts())

    dataset = KDEFTripletDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    E = GeometryAutoEncoder(input_dim=136, z_g_dim=Z_G_DIM).to(DEVICE)
    G = Generator(z_i_dim=Z_I_DIM, z_g_dim=Z_G_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)

    train_E(E, loader)
    train_GAN(E, G, D, loader)


if __name__ == "__main__":
    main()