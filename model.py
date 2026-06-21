import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryAutoEncoder(nn.Module):
    def __init__(self, input_dim=136, z_g_dim=32):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Linear(64, z_g_dim),
            nn.BatchNorm1d(z_g_dim),
            nn.ReLU(True),
        )

        self.dec = nn.Sequential(
            nn.Linear(z_g_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, input_dim),
            nn.Tanh(),
        )

    def forward(self, g):
        z_g = self.enc(g)
        g_hat = self.dec(z_g)
        return z_g, g_hat


class FiLM(nn.Module):
    def __init__(self, z_dim, channels):
        super().__init__()

        self.net = nn.Linear(z_dim, channels * 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, x, z):
        gamma_beta = self.net(z)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return x * (1.0 + gamma) + beta


class GN(nn.Module):
    def __init__(self, ch, max_groups=8):
        super().__init__()

        groups = min(max_groups, ch)
        while ch % groups != 0:
            groups -= 1

        self.norm = nn.GroupNorm(groups, ch)

    def forward(self, x):
        return self.norm(x)


class ResBlockNoNorm(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = F.silu(h)

        h = self.conv2(h)

        return F.silu(h + self.skip(x))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm1 = GN(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm2 = GN(out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return F.silu(h + self.skip(x))


class ResBlockFiLMNoNorm(nn.Module):
    def __init__(self, in_ch, out_ch, z_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.film1 = FiLM(z_dim, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.film2 = FiLM(z_dim, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, z):
        h = self.conv1(x)
        h = self.film1(h, z)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.film2(h, z)

        return F.silu(h + self.skip(x))


class ResBlockFiLM(nn.Module):
    def __init__(self, in_ch, out_ch, z_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm1 = GN(out_ch)
        self.film1 = FiLM(z_dim, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm2 = GN(out_ch)
        self.film2 = FiLM(z_dim, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, z):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film1(h, z)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.film2(h, z)

        return F.silu(h + self.skip(x))


class SpatialTransformer(nn.Module):
    def __init__(self, channels, num_heads=4, depth=1):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )

        self.norm = nn.LayerNorm(channels)

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
        )

    def forward(self, x):
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        tokens = self.transformer(tokens)

        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x


class DownNoNorm(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.down = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        self.block = ResBlockNoNorm(out_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.down = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        self.block = ResBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x


class UpNoNorm(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, z_dim):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.block = ResBlockFiLMNoNorm(out_ch + skip_ch, out_ch, z_dim)

    def forward(self, x, skip, z):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, z)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, z_dim):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.block = ResBlockFiLM(out_ch + skip_ch, out_ch, z_dim)

    def forward(self, x, skip, z):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, z)
        return x


class Generator(nn.Module):
    def __init__(self, z_i_dim=128, z_g_dim=32):
        super().__init__()

        z_dim = 128

        self.z_proj = nn.Sequential(
            nn.Linear(z_g_dim, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
            nn.SiLU(),
        )

        self.in_conv = nn.Conv2d(3, 32, 3, 1, 1)

        # Đầu mạng: NoNorm để giữ màu/ánh sáng nguồn
        self.enc0 = ResBlockNoNorm(32, 32)       # 128 x 128
        self.down1 = DownNoNorm(32, 64)          # 64 x 64

        # Giữa mạng: GroupNorm để ổn định training
        self.down2 = Down(64, 128)               # 32 x 32
        self.down3 = Down(128, 256)              # 16 x 16
        self.down4 = Down(256, 512)              # 8 x 8

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            GN(512),
            nn.SiLU(),
        )

        self.transformer = SpatialTransformer(
            channels=512,
            num_heads=4,
            depth=1,
        )

        self.mid = ResBlockFiLM(512, 512, z_dim)

        # Giữa decoder: GroupNorm
        self.up4 = Up(512, 256, 256, z_dim)      # 16 x 16
        self.up3 = Up(256, 128, 128, z_dim)      # 32 x 32

        # Cuối decoder: NoNorm để khôi phục màu RGB tốt hơn
        self.up2 = UpNoNorm(128, 64, 64, z_dim)  # 64 x 64
        self.up1 = UpNoNorm(64, 32, 32, z_dim)   # 128 x 128

        self.out = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, img, z_g):
        z = self.z_proj(z_g)

        x0 = self.in_conv(img)
        s0 = self.enc0(x0)

        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)

        x = self.bottleneck(x)
        x = self.transformer(x)
        x = self.mid(x, z)

        x = self.up4(x, s3, z)
        x = self.up3(x, s2, z)
        x = self.up2(x, s1, z)
        x = self.up1(x, s0, z)

        return self.out(x)






class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()

        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
        ]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        self.net = nn.Sequential(
            DiscBlock(in_ch, base_ch, use_norm=False),      # 128 -> 64
            DiscBlock(base_ch, base_ch * 2),                # 64 -> 32
            DiscBlock(base_ch * 2, base_ch * 4),            # 32 -> 16
            DiscBlock(base_ch * 4, base_ch * 8),            # 16 -> 8

            nn.utils.spectral_norm(
                nn.Conv2d(base_ch * 8, base_ch * 8, 3, 1, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(
                nn.Conv2d(base_ch * 8, 1, kernel_size=3, stride=1, padding=1)
            )                                               # 8 -> 8 patch map
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0), -1)


if __name__ == "__main__":
    img = torch.randn(4, 3, 128, 128)
    g = torch.randn(4, 136)

    E = GeometryAutoEncoder(input_dim=136, z_g_dim=32)
    G = Generator(z_i_dim=128, z_g_dim=32)
    D = Discriminator()

    z_g, g_hat = E(g)
    fake = G(img, z_g)
    score = D(fake)

    print("z_g:", z_g.shape)
    print("g_hat:", g_hat.shape)
    print("fake:", fake.shape)
    print("D:", score.shape)