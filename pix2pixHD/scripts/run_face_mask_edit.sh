set -xue

# test global network
LOAD_SIZE=512
NETG="global"
PRETRAIN_MODEL="./CelebA_pretrain/epoch_33_iter_3900_Gloss_11.698519_Dloss_0.452646"


python3 face_mask_edit.py \
    --loadSize $LOAD_SIZE \
    --netG $NETG \
    --load_pretrain $PRETRAIN_MODEL
