import os
import shutil

local_params_dir = "checkpoint_local_CelebA_only/epoch_5_iter_13800_Gloss_8.562819_Dloss_0.007365/"
global_params_dir = "checkpoint_global_CelebA/epoch_17_iter_8400_Gloss_9.519454_Dloss_1.200356/"
result_dir = "./checkpoint_local_CelebA/merge_local_only_epoch_5_global_epoch_17/"

local_params = os.listdir(local_params_dir)
global_params = os.listdir(global_params_dir)

all_params = []
for k in local_params:
    all_params.append((os.path.join(local_params_dir, k), k))

for k in global_params:
    if k not in local_params:
        all_params.append((os.path.join(global_params_dir, k), k))


for k in all_params:
    if os.path.isfile(k[0]):
        shutil.copy(k[0], os.path.join(result_dir, k[1]))
    else:
        shutil.copytree(k[0], os.path.join(result_dir, k[1]))
