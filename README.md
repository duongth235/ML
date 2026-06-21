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

The implementation is trained on the **KDEF** facial expression dataset and uses:

* Geometry AutoEncoder
* FiLM-conditioned Generator
* Patch-based WGAN-GP Discriminator
* Landmark-based expression representation
* ArcFace-based identity evaluation
* Expression-classifier-based expression evaluation

---

# Project Structure

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
├── validation/
│   │
│   ├── validation_generate.py
│   ├── gen_data/
│   │
│   ├── id_validation/
│   │   ├── build_gallery.py
│   │   ├── evaluate_id.py
│   │   ├── gallery_embeddings.npz
│   │   └── ...
│   │
│   └── expression_validation/
│       ├── classifier_train.py
│       ├── evaluate_expression.py
│       └── ...
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

Dataset statistics:

```text
110 identities
7 expressions
770 frontal images
```

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
* reconstructed geometry ĝ

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
10GP
```

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

### Color Preservation Loss

```math
L_{color}
=
L1(mean(fake),mean(source))
```

### Identity Preservation Loss

```math
L_{id}
=
L1(fake_{self}, source)
```

### Final Generator Loss

```math
L_G
=
λ_{ir}L_{ir}
+
λ_{adv}L_{adv}
+
λ_{color}L_{color}
+
λ_{id}L_{id}
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

Result:

```text
[source | target | generated]
```

saved to:

```text
outputs_gcgan/generated_result.jpg
```

---

# Validation and Evaluation

The project includes a validation pipeline for evaluating both identity preservation and expression transfer.

Generated validation images:

```text
validation/gen_data/
```

Filename format:

```text
sourceID_expression_index.png
```

Example:

```text
12_happy_003.png
```

---

## Generate Validation Samples

```bash
python validation/validation_generate.py
```

Generates:

```text
30 samples per expression
210 generated images total
```

---

## Identity Evaluation

### Build ArcFace Gallery

```bash
python validation/id_validation/build_gallery.py
```

Creates:

```text
validation/id_validation/gallery_embeddings.npz
```

Each identity is represented by the average ArcFace embedding of its seven frontal expression images.

### Evaluate Identity Preservation

```bash
python validation/id_validation/evaluate_id.py
```

Metrics:

* Mean ID Cosine Similarity
* Rank-1 Accuracy
* Rank-3 Accuracy

Results:

```text
Mean ID Cosine : 0.6481
Rank-1 Accuracy: 92.86%
Rank-3 Accuracy: 100.00%
```

---

## Expression Classifier Training

Train a classifier used only for evaluation.

```bash
python validation/expression_validation/classifier_train.py
```

Model:

```text
ResNet18
ImageNet Pretrained
7 Expression Classes
```

Checkpoint:

```text
validation/expression_validation/expression_classifier_best.pth
```

Validation performance on real KDEF:

```text
Validation Accuracy ≈ 94.8%
Validation Macro F1 ≈ 94.7%
```

---

## Expression Evaluation

Evaluate generated images:

```bash
python validation/expression_validation/evaluate_expression.py
```

Metrics:

* Expression Accuracy
* Expression Macro F1
* Per-Class Accuracy
* Confusion Matrix

Results:

```text
Expression Accuracy : 66.19%
Expression Macro F1 : 63.70%
```

Generated files:

```text
validation/expression_validation/expression_eval_results.csv
validation/expression_validation/expression_confusion_matrix.csv
validation/expression_validation/expression_per_class_accuracy.csv
```

---

# Evaluation Summary

| Category   | Metric          | Value   |
| ---------- | --------------- | ------- |
| Identity   | Mean ID Cosine  | 0.6481  |
| Identity   | Rank-1 Accuracy | 92.86%  |
| Identity   | Rank-3 Accuracy | 100.00% |
| Expression | Accuracy        | 66.19%  |
| Expression | Macro F1        | 63.70%  |

The model preserves identity effectively while expression transfer remains more challenging, particularly for subtle expressions such as `sad` and `neutral`.

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

## Geometry

* MediaPipe FaceMesh (478 landmarks)
* 3D landmarks

## Identity

* ArcFace identity loss
* InsightFace identity encoder

## Generator

* Cross-Attention conditioning
* Style-based decoder

## Diffusion Models

Replace GAN with:

* Stable Diffusion
* DiT
* Conditional Flow Matching
* Rectified Flow

for higher-fidelity expression transfer.

---

# References

1. GC-GAN: Geometry-Contrastive GAN for Facial Expression Transfer
2. KDEF Dataset
3. FiLM: Feature-wise Linear Modulation
4. WGAN-GP
5. U-Net
6. Spatial Transformer Networks
7. ArcFace
8. InsightFace
