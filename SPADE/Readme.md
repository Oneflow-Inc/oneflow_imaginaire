# Implementation of [SPAED](https://arxiv.org/abs/1903.07291) with [Oneflow](https://github.com/Oneflow-inc/oneflow) framework.

##Results
###Cityscapes
<div align='center'>
  <img src='results/inference0_.jpg'>
</div>

<div align='center'>
  <img src='results/inference1_.jpg'>
</div>

<div align='center'>
  <img src='results/inference2_.jpg'>
</div>

<div align='center'>
  <img src='results/inference3_.jpg'>
</div>

<div align='center'>
  <img src='results/inference4_.jpg'>
</div>

## Requirements

- python3
- [Oneflow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)

## Training

Download the VGG16 pretrained model:
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/vgg16_of_best_model_val_top1_721.zip
```
### Training on Cityscapes task

#### 1. Train 0-100 epoch
```bash
python train.py --batch_size 8 --gpu_nums 8 --pre_vgg ./vgg16_of_best_model_val_top1_721 --beta1 0.5 --beta2 0.999 --lr_D 2e-5 --lr_G 2e-5 --up nearest --lambda_vgg 100
```

#### 2. Train 100-200 epoch, using pre-trained model as above
```bash
python train.py --batch_size 8 --gpu_nums 8 --pre_vgg ./vgg16_of_best_model_val_top1_721 --beta1 0.5 --beta2 0.999 --lr_D 2e-5 --lr_G 2e-5 --up nearest --lambda_vgg 100 --pre_G_D epoch_99iter_|3002021-08-08-19-36-54
```