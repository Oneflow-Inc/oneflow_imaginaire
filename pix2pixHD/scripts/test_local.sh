set -xue

# test global network
LOAD_SIZE=2048
GPU_NUMS=1
BATCH_SIZE=1
NETG="local"
NGF=32
DATA_ROOT="./datasets/cityscapes"
# PRETRAIN_MODEL="./checkpoints_local/epoch_49_iter_0_Gloss_5.615729_Dloss_0.677637"
PRETRAIN_MODEL="./checkpoints_local/epoch_50_iter_2100_Gloss_5.392593_Dloss_0.694629"
RESULT_DIR="results_local_50"

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
