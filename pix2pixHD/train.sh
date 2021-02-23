set -xue

#### train pix2pixHD first stage on single device
#### run this scirpt with `CUDA_VISIBLE_DEVICES=1 bash train.sh`

LOAD_SIZE=1024
GPU_NUMS=1
BATCH_SIZE=1
NETG="global"
DATA_ROOT="./cityscapes_pix2pixHD"
LOAD_PRETRAIN="./checkpoints/opech_31_iter_1500_Gloss_6.100729_Dloss_0.753867"

python3 train_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --dataroot $DATA_ROOT \
    --load_pretrain $LOAD_PRETRAIN

#### train pix2pixHD second stage on single device
#### you will get out of memory message if your device memory is less than 24G
#### run this scirpt with `CUDA_VISIBLE_DEVICES=1 bash train.sh`
# python3 train_of_pix2pixhd.py --loadSize 2048 --gpu_nums 1 --batchSize 1 --netG "local"


#### train pix2pixHD second stage with pipeline parallelism on 4 devices
#### run this scirpt with `CUDA_VISIBLE_DEVICES=0,1,2,3 bash train.sh`
# python3 train_of_pix2pixhd_pipeline_parallel.py --loadSize 2048 --gpu_nums 4 --batchSize 1 --netG "local"

