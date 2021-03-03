set -xue

#### train pix2pixHD full generative networks (local + global) 
#### with pipeline parallelism on 3 devices with device memory bigger than 10G.
#### run this scirpt with `CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_local_CelebA.sh`

LOAD_SIZE=1024
GPU_NUMS=2
BATCH_SIZE=1
NETG="local"
NGF=32
DATA_ROOT="/DATA/disk1/ldp/CelebAMask-HQ"
LABEL_NC=19
LR=0.00002
TRAIN_TMP_RESULT="train_CelebA_local_tmp_result.jpg"
CHECKPOINT_DIR="checkpoint_local_CelebA"
LOAD_PRETRAIN="checkpoint_local_CelebA/merge_local_only_epoch_5_global_epoch_17"

if [ ! -d $CHECKPOINT_DIR ] ; then
  mkdir -p $CHECKPOINT_DIR
fi

python3 train_of_pix2pixhd_pipeline_parallel.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --ngf $NGF \
    --no_instance \
    --lr $LR \
    --dataroot $DATA_ROOT \
    --train_tmp_result $TRAIN_TMP_RESULT \
    --label_nc $LABEL_NC \
    --checkpoints_dir $CHECKPOINT_DIR \
    --load_pretrain $LOAD_PRETRAIN
    #--no_global_generator   ### if you want to train local network only, 
                              ### you can pass this parameter into the
                              ### `train_of_pix2pixhd_pipeline_parallel.py` script

