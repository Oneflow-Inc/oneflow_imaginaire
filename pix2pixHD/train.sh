
#### train pix2pixHD first stage on single device
#### run this scirpt with `CUDA_VISIBLE_DEVICES=1 bash train.sh`
# python3 train_of_pix2pixhd.py --loadSize 1024 --gpu_nums 1 --batchSize 1 --netG "global"


#### train pix2pixHD second stage on single device
#### you will get out of memory message if your device memory is less than 24G
#### run this scirpt with `CUDA_VISIBLE_DEVICES=1 bash train.sh`
# python3 train_of_pix2pixhd.py --loadSize 2048 --gpu_nums 1 --batchSize 1 --netG "local"


#### train pix2pixHD second stage with pipeline parallelism on 4 devices
#### run this scirpt with `CUDA_VISIBLE_DEVICES=0,1,2,3 bash train.sh`
python3 train_of_pix2pixhd_pipeline_parallel.py --loadSize 2048 --gpu_nums 4 --batchSize 1 --netG "local"

