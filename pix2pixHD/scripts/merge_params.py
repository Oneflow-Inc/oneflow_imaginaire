import os
import shutil

local_params_dir = "checkpoints_local_only/epoch_8_iter_900_Gloss_9.246778_Dloss_0.012707/"
global_params_dir = "checkpoints/epoch_78_iter_2100_Gloss_7.245208_Dloss_0.941353/"
result_dir = "./checkpoints_local/merge_local_only_epoch_8_global_epoch_78/"

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
