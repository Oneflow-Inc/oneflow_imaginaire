set -xue

# test global network
LOAD_SIZE=512
NETG="global"
PRETRAIN_MODEL="./CelebAHQ_global"


python3 face_mask_edit.py \
    --loadSize $LOAD_SIZE \
    --netG $NETG \
    --load_pretrain $PRETRAIN_MODEL
