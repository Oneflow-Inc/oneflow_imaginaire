set -xue

# test global network
LOAD_SIZE=2048
GPU_NUMS=1
BATCH_SIZE=1
NETG="local"
NGF=32
DATA_ROOT="./datasets/cityscapes"
PRETRAIN_MODEL="./Cityscapes_pretrain/epoch_194_iter_600_Gloss_6.874650_Dloss_0.702645"
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
