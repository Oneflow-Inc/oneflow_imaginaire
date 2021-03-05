set -xue

# test global network
LOAD_SIZE=2048
GPU_NUMS=1
BATCH_SIZE=1
NETG="local"
NGF=32
DATA_ROOT="./datasets/cityscapes"
PRETRAIN_MODEL="./Cityscapes_pretrain/epoch_56_iter_1200_Gloss_5.284343_Dloss_0.841812"
RESULT_DIR="results_local_cityscapes"

if [ ! -d $RESULT_DIR ] ; then
  mkdir -p $RESULT_DIR
fi

python3 test_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --ngf $NGF \
    --dataroot $DATA_ROOT \
    --load_pretrain $PRETRAIN_MODEL \
    --results_dir $RESULT_DIR
