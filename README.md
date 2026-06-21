# Geometry-Conditioned GAN for Facial Expression Transfer

## Overview

This project implements a **Geometry-Conditioned GAN (GC-GAN)** for facial expression transfer using facial landmarks as geometric guidance.

Given:

* A **source face image** (identity to preserve)
* A **target expression geometry** extracted from another face

The model generates a new face image that:

* Preserves the identity of the source person
* Transfers the target facial expression
* Maintains realistic appearance and facial structure

The implementation is trained on the KDEF facial expression dataset and uses:

* Geometry AutoEncoder
* FiLM-conditioned Generator
* Patch-based WGAN-GP Discriminator
* Landmark-based expression representation

---

## Project Structure

```text
project/
│
├── config.py
├── model.py
├── train.py
├── generate.py
│
├── datasets/
│   └── processed_KDEF/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
│
└── outputs_gcgan/
    ├── landmarks_front_pose.csv
    ├── E.pth
    ├── G_xxx.pth
    ├── D_xxx.pth
    ├── sample_epoch_xxx.jpg
    └── generated_result.jpg
```

---

# Dataset

## KDEF

The project uses the KDEF dataset.

Expressions:

* angry
* disgust
* fear
* happy
* neutral
* sad
* surprise

Only frontal images are used:

```text
*_2.jpg
```

Example:

```text
000_happy_2.jpg
000_sad_2.jpg
000_angry_2.jpg
```

where:

* identity = 000
* expression = happy/sad/angry
* pose = frontal

---

# Landmark Representation

Facial geometry is represented using:

* 68 facial landmarks
* 2D coordinates

Dimension:

```text
68 × 2 = 136
```

Coordinates are normalized to:

```text
[-1, 1]
```

using:

```python
x = 2*x/w - 1
y = 2*y/h - 1
```

Landmarks are extracted using:

```python
face_alignment
```

and cached into:

```text
outputs_gcgan/landmarks_front_pose.csv
```

---

# Model Architecture

## 1. Geometry AutoEncoder

Input:

```text
g ∈ R^136
```

Encoder:

```text
136
 → 128
 → 64
 → z_g
```

Default:

```text
z_g = 32
```

Decoder:

```text
32
 → 64
 → 128
 → 136
```

Outputs:

* latent expression embedding z_g
* reconstructed geometry ĝ

---

## 2. Generator

The generator is a U-Net style architecture.

Input:

```text
source image
+
expression embedding z_g
```

Geometry conditioning is injected through:

```text
FiLM layers
```

Architecture:

```text
Image
  ↓
Encoder
  ↓
Transformer Bottleneck
  ↓
Decoder + FiLM
  ↓
Generated Face
```

Features:

* Skip connections
* FiLM conditioning
* Spatial Transformer bottleneck
* GroupNorm in deep layers
* NoNorm blocks near image space to preserve color

---

## 3. Discriminator

PatchGAN discriminator with:

```text
Spectral Normalization
```

and

```text
Instance Normalization
```

Objective:

```text
WGAN-GP
```

Output:

```text
Patch realism scores
```

instead of a single scalar score.

---

# Training Pipeline

Training consists of two stages.

---

## Stage 1: Geometry AutoEncoder Pretraining

The geometry encoder learns expression embeddings.

Loss:

```math
L_E
=
λ_contr L_contrastive
+
λ_gr L_reconstruction
```

### Contrastive Loss

Same expression:

```math
||z_1-z_2||^2
```

Different expression:

```math
max(margin - ||z_1-z_2||^2,0)
```

### Geometry Reconstruction

```math
MSE(g,\hat g)
```

---

## Stage 2: GAN Training

Freeze:

```text
Geometry Encoder E
```

Train:

```text
Generator G
Discriminator D
```

---

### Adversarial Loss

WGAN-GP:

```math
L_G^{adv}
=
-D(G(x))
```

```math
L_D
=
D(fake)
-
D(real)
+
10 GP
```

---

### Image Reconstruction Loss

RGB reconstruction:

```math
L_{rgb}
=
L1(fake,target)
```

Instance-normalized reconstruction:

```math
L_{norm}
=
L1(IN(fake),IN(target))
```

Combined:

```math
L_{ir}
=
λ_{rgb}L_{rgb}
+
λ_{norm}L_{norm}
```

---

### Color Preservation Loss

```math
L_{color}
=
L1(mean(fake),mean(source))
```

Preserves overall illumination and color tone.

---

### Identity Preservation Loss

Self-reconstruction:

```math
G(source, source_geometry)
≈ source
```

Loss:

```math
L_{id}
=
L1(fake_self, source)
```

---

### Final Generator Loss

```math
L_G
=
λ_ir L_ir
+
λ_adv L_adv
+
λ_color L_color
+
λ_id L_id
```

---

# Default Hyperparameters

```python
IMG_SIZE = 128

BATCH_SIZE = 32

EPOCHS_E = 80
EPOCHS_GAN = 140

LR = 3e-4

LAMBDA_CONTR = 1.0
LAMBDA_GR    = 1.0

LAMBDA_IR    = 3.0
LAMBDA_ADV   = 8e-5
LAMBDA_COLOR = 1.2
LAMBDA_ID    = 1.7

LAMBDA_RGB   = 0.5
LAMBDA_NORM  = 0.5

MARGIN = 6.0
```

---

# Training

Run:

```bash
python train.py
```

Training flow:

```text
Detect Landmarks
      ↓
Create Cache
      ↓
Train Geometry AutoEncoder
      ↓
Freeze Encoder
      ↓
Train GC-GAN
      ↓
Save Checkpoints
```

Outputs:

```text
outputs_gcgan/
```

---

# Inference

Random expression transfer:

```bash
python generate.py
```

Specific source and target:

```bash
python generate.py \
    --source path/to/source.jpg \
    --target path/to/target.jpg
```

Result:

```text
[source | target | generated]
```

saved to:

```text
outputs_gcgan/generated_result.jpg
```

---

# Example

Source:

```text
Identity A
Expression: Neutral
```

Target:

```text
Identity B
Expression: Happy
```

Generated:

```text
Identity A
Expression: Happy
```

---

# Future Improvements

Potential upgrades:

### Geometry

* MediaPipe FaceMesh (478 landmarks)
* 3D landmarks

### Identity

* ArcFace embedding
* InsightFace identity loss

### Generator

* Cross-Attention conditioning
* Style-based decoder

### Diffusion Models

Replace GAN with:

* Stable Diffusion
* DiT
* Conditional Flow Matching
* Rectified Flow

for higher fidelity expression transfer.

---

# References

1. GC-GAN: Geometry-Contrastive GAN for Facial Expression Transfer

2. KDEF Dataset

3. FiLM: Feature-wise Linear Modulation

4. WGAN-GP

5. U-Net

6. Spatial Transformer Networks
# ML
