# Rethinking Multi-Modal Fusion: A Differential and Entropy-driven Network for 3D Brain Tumor Segmentation
## Usage
### Data Preparation
Please download BraTS 2020 data according to https://www.med.upenn.edu/cbica/brats2020/data.html.
### Training
#### Training on the entire BraTS training set
```bash
python train.py --model DENet --mixed --trainset --gpu 3 --dataset brats2020
```
#### Breakpoint continuation for training
```bash
python train.py --model DENet --mixed --trainset --cp checkpoint
```
### Inference
```bash
python test.py --model DENet --tta --labels --post_process --save_pred --cp checkpoint
```
