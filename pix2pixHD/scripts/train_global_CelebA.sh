set -xue

#### train pix2pixHD global network on single device with device memory bigger than 8G.
#### run this scirpt with `CUDA_VISIBLE_DEVICES=1 bash scripts/train_global_CelebA.sh`
LOAD_SIZE=512
GPU_NUMS=1
BATCH_SIZE=1
NETG="global"
DATA_ROOT="./datasets/CelebA_HQ"
LABEL_NC=19
LR=0.00002
TRAIN_TMP_RESULT="train_CelebA_global_tmp_result.jpg"
CHECKPOINT_DIR="checkpoint_global_CelebA"
LOAD_PRETRAIN="vgg16_of_best_model_val_top1_721"

if [ ! -d $CHECKPOINT_DIR ] ; then
  mkdir -p $CHECKPOINT_DIR
fi

python3 train_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --no_instance \
    --lr $LR \
    --dataroot $DATA_ROOT \
    --train_tmp_result $TRAIN_TMP_RESULT \
    --label_nc $LABEL_NC \
    --checkpoints_dir $CHECKPOINT_DIR \
    --load_pretrain $LOAD_PRETRAIN


