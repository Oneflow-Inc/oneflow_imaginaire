set -xue

# test global network
LOAD_SIZE=1024
GPU_NUMS=1
BATCH_SIZE=1
NETG="global"
DATA_ROOT="./datasets/cityscapes"
PRETRAIN_MODEL="./checkpoints/opech_94_iter_1500_Gloss_6.976357_Dloss_0.474898"
RESULT_DIR="results"

if [ ! -d $RESULT_DIR ] ; then
  mkdir -p $RESULT_DIR
fi

python3 test_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --dataroot $DATA_ROOT \
    --load_pretrain $PRETRAIN_MODEL \
    --results_dir $RESULT_DIR


