set -xue

IMAGE_SIZE=512
BATCH_SIZE=1
LABEL_NC=119
LR=0.0001

python3 train_of_pix2pixhd.py \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --label_nc $LABEL_NC \
    --lr $LR
