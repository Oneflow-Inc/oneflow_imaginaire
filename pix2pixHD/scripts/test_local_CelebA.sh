set -xue

# test global network
LOAD_SIZE=1024
GPU_NUMS=1
BATCH_SIZE=1
NETG="local"
NGF=32
DATA_ROOT="/DATA/disk1/ldp/CelebAMask-HQ"
LABEL_NC=19
PRETRAIN_MODEL="./checkpoint_local_CelebA/epoch_0_iter_12300_Gloss_9.614251_Dloss_0.544747"
RESULT_DIR="results_local_CelebA"

if [ ! -d $RESULT_DIR ] ; then
  mkdir -p $RESULT_DIR
fi

python3 test_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --ngf $NGF \
    --no_instance \
    --label_nc $LABEL_NC \
    --dataroot $DATA_ROOT \
    --load_pretrain $PRETRAIN_MODEL \
    --results_dir $RESULT_DIR
