# Implementation of [pix2pixHD](https://arxiv.org/pdf/1711.11585.pdf) with [Oneflow](https://github.com/Oneflow-inc/oneflow) framework.

This work is based on the repo: [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ).


## Some Results

### Cityscapes
First train the global net for 78 epoches. And then jointly train the local enhancer and global net for 56 epoches:

<div align='center'>
  <img src='results/cityscapes.png'>
</div>


### CelebA-HQ

Only train the global nets for 33 epoches:
<div align='center'>
  <img src='results/CelebAMask-HQ.png '>
</div>



### Demo of interactive facial image manipulation

screenshots
<img src="results/demo.gif"/>


## Tested on
| Spec                        |                                                             |
|-----------------------------|-------------------------------------------------------------|
| GPU                         | Nvidia TITAN V                                              |
| CUDA Version                | 11.1                                                        |
| Driver Version              | 455.32.00                                                   |


## Requirements

- python3
- [Oneflow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)
- `pip3 install -r requirements.txt`

## Testing
### Pretrain models

Download the `Cityscapes` and `CelebAMask-HQ` pretrain models with:

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cv/gan/pix2pixHD_pertrain_model.tar.gz
```

### Testing on Cityscapes or CelebAMask-HQ

Set the downloaded pretrain model path on scripts `scripts/test_local_cityscapes.sh` or `scripts/test_global_CelebA.sh` and run:

```
PRETRAIN_MODEL=""
```

### Run demo of interactive facial image manipulation

Also set the `PRETRAIN_MODEL` path in script `scripts/run_face_mask_edit.sh` and run.
## Training

First download the VGG16 pretrain model:
```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/vgg16_of_best_model_val_top1_721.zip
```
### Training on Cityscapes task

#### First, train the global net and local enhancer seperately

Train global net, remember to set the right VGG16 model path:
```bash
bash scripts/train_global_cityscapes.sh
```

Train local enhancer only, first uncomment the last line of script `scripts/train_local_cityscapes.sh`:

```bash
python3 train_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --ngf $NGF \
    --lr $LR \
    --network_module $MODULE \
    --dataroot $DATA_ROOT \
    --checkpoints_dir $CHECKPOINT_DIR \
    --train_tmp_result $TRAIN_TMP_RESULT \
    --load_pretrain $LOAD_PRETRAIN \
    --no_global_generator     ### if you want to train local network only, 
                              ### you can pass this parameter into the
                              ### `train_of_pix2pixhd_pipeline_parallel.py` script
```

to make sure only train the local enhancer.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_local_cityscapes.sh
```

#### Second, jointly train the local enhancer and global net

first, merge the local and global checkpoints together with script:

 ```bash
python3 scripts/merge_params.py
 ```
 
and then comment the last line of script `scripts/train_local_cityscapes.sh` and set the `LOAD_PRETRAIN` to the merged checkpoint path:

```bash
python3 train_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --ngf $NGF \
    --lr $LR \
    --network_module $MODULE \
    --dataroot $DATA_ROOT \
    --checkpoints_dir $CHECKPOINT_DIR \
    --train_tmp_result $TRAIN_TMP_RESULT \
    --load_pretrain $LOAD_PRETRAIN \
    # --no_global_generator     ### if you want to train local network only, 
                              ### you can pass this parameter into the
                              ### `train_of_pix2pixhd_pipeline_parallel.py` script
```

last, make sure that each GPU has at least 10G memory and run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_local_cityscapes.sh
```

### Training on CelebAMask-HQ task

```bash
bash scripts/train_global_CelebA.sh
```



