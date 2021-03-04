set -xue

# test global network
LOAD_SIZE=512
GPU_NUMS=1
BATCH_SIZE=1
NETG="global"
DATA_ROOT="/home/ldpe2g/oneFlow/OtherProjects/CelebAMask-HQ/MaskGAN_demo/CelebA_HQ"
LABEL_NC=19
PRETRAIN_MODEL="./CelebAHQ_global"
RESULT_DIR="results_CelebA"


if [ ! -d $RESULT_DIR ] ; then
  mkdir -p $RESULT_DIR
fi

python3 test_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --no_instance \
    --label_nc $LABEL_NC \
    --dataroot $DATA_ROOT \
    --load_pretrain $PRETRAIN_MODEL \
    --results_dir $RESULT_DIR
