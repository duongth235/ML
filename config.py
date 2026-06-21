from pathlib import Path

DATA_ROOT = Path("datasets/processed_KDEF")
OUT_DIR = Path("outputs_gcgan")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_E = 80
EPOCHS_GAN = 140

LR = 3e-4
BETA1 = 0.5
BETA2 = 0.999

LAMBDA_CONTR = 1.0
LAMBDA_GR    = 1.0

LAMBDA_IR    = 3.0
LAMBDA_ADV   = 8e-5
LAMBDA_COLOR = 1.2
LAMBDA_ID    = 1.7

LAMBDA_RGB   = 0.5
LAMBDA_NORM  = 0.5

MARGIN = 6.0

Z_G_DIM = 32
Z_I_DIM = 224

NUM_WORKERS = 0
DEVICE = "mps"   # đổi thành "cuda" nếu dùng NVIDIA
EXPAND_RATIO = 1.25
