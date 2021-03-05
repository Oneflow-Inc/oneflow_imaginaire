set -xue

#### train pix2pixHD full generative networks (local + global) 
#### with pipeline parallelism on 3 devices with device memory bigger than 10G.
#### run this scirpt with `CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_local_cityscapes.sh`

LOAD_SIZE=2048
GPU_NUMS=3
BATCH_SIZE=1
NETG="local"
NGF=32
LR=0.0000002
MODULE="models.networks_pipeline_parallel"
DATA_ROOT="./datasets/cityscapes"
CHECKPOINT_DIR="checkpoints_local"
TRAIN_TMP_RESULT="train_cityscapes_local_tmp_result.jpg"
LOAD_PRETRAIN="vgg16_of_best_model_val_top1_721"

if [ ! -d $CHECKPOINT_DIR ] ; then
  mkdir -p $CHECKPOINT_DIR
fi

python3 train_of_pix2pixhd.py \
    --loadSize $LOAD_SIZE \
    --gpu_nums $GPU_NUMS \
    --batchSize $BATCH_SIZE \
    --netG $NETG \
    --ngf $NGF \
    --lr $LR \
    --network_module $MODULE \
    --dataroot $DATA_ROOT \
    --checkpoints_dir $CHECKPOINT_DIR \
    --train_tmp_result $TRAIN_TMP_RESULT \
    --load_pretrain $LOAD_PRETRAIN
    # --no_global_generator   ### if you want to train local network only, 
                              ### you can pass this parameter into the
                              ### `train_of_pix2pixhd_pipeline_parallel.py` script

