set -xue

# test global network
LOAD_SIZE=2048
GPU_NUMS=1
BATCH_SIZE=1
NETG="local"
NGF=32
DATA_ROOT="./datasets/cityscapes"
# PRETRAIN_MODEL="./checkpoints_local/epoch_39_iter_1500_Gloss_5.033952_Dloss_0.704229"
PRETRAIN_MODEL="./checkpoints_local/epoch_41_iter_2400_Gloss_6.147730_Dloss_0.699599"
RESULT_DIR="results_local"

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
