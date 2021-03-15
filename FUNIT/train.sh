set -xue

NUM_GPUS=1 # 1 or 2
IMAGE_SIZE=512
BATCH_SIZE=1
LABEL_NC=119
LR=0.0001

python3 train_of_pix2pixhd.py \
    --num_gpus $NUM_GPUS \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --label_nc $LABEL_NC \
    --lr $LR
