set -xue

#### train pix2pixHD global network on single device with device memory bigger than 8G.
#### run this scirpt with `CUDA_VISIBLE_DEVICES=1 bash scripts/train_global_CelebA.sh`
LOAD_SIZE=512
GPU_NUMS=1
BATCH_SIZE=1
NETG="global"
DATA_ROOT="/DATA/disk1/ldp/CelebAMask-HQ"
LABEL_NC=19
TRAIN_TMP_RESULT="train_CelebA_global_tmp_result.jpg"
CHECKPOINT_DIR="checkpoint_global_CelebA"
LOAD_PRETRAIN="checkpoint_global_CelebA/epoch_11_iter_23100_Gloss_10.468245_Dloss_0.149702"

if [ ! -d $CHECKPOINT_DIR ] ; then
  mkdir -p $CHECKPOINT_DIR
fi

python3 train_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --no_instance \
    --dataroot $DATA_ROOT \
    --train_tmp_result $TRAIN_TMP_RESULT \
    --label_nc $LABEL_NC \
    --checkpoints_dir $CHECKPOINT_DIR \
    --load_pretrain $LOAD_PRETRAIN


