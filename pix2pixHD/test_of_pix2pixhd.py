import os
import numpy as np
import argparse
import cv2

import oneflow as flow
import oneflow.typing as tp

import models.networks as networks
import util.util as util
import util.image_pool as image_pool

from data.aligned_dataset import AlignedDataset
from options.test_options import TestOptions

opt = TestOptions().parse()

device_type = "gpu" if opt.gpu_nums > 0 else "cpu"
if device_type == "gpu":
    flow.config.gpu_device_num(opt.gpu_nums)

flow.env.init()
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_logical_view(flow.scope.consistent_view())
func_config.default_placement_scope(flow.scope.placement("gpu", "0:0"))

# load dataset
dataset = AlignedDataset()
print("dataset [%s] was created" % (dataset.name()))
dataset.initialize(opt)

batch, channel, height, width = dataset[0]["label"].shape

label_class_num = opt.label_nc
if not opt.no_instance:
    label_class_num += 1

@flow.global_function("predict", func_config)
def TrainGenerators(
    label: tp.Numpy.Placeholder((opt.batchSize, label_class_num, height, width), dtype = flow.float32),):
    fake_image = networks.define_G(label, opt.output_nc, opt.ngf, opt.netG,
                                n_downsample_global=opt.n_downsample_global, n_blocks_global=opt.n_blocks_global,
                                n_blocks_local=opt.n_blocks_local, norm_type=opt.norm, trainable=False, reuse=True)
    return fake_image


dataset_len = len(dataset)

assert opt.load_pretrain != ""
flow.load_variables(flow.checkpoint.get(opt.load_pretrain))

if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

for i in range(dataset_len):
    data_dict = dataset[i]

    label_one_hot_encoding = np.zeros((batch, opt.label_nc, height, width), dtype=np.float)
    util.scatter(label_one_hot_encoding, 1, data_dict['label'].astype(np.int32), 1)
    label_nd = label_one_hot_encoding
    if not opt.no_instance:
        edge_nd = util.get_inst_map_edge(data_dict["inst"].astype(np.int32))
        label_nd = np.concatenate((label_nd, edge_nd.astype(np.float)), axis = 1)
    
    fake_image = TrainGenerators(label_nd).get()

    fake_image = util.tensor2im(fake_image.numpy()[0])
    label = util.onehot2label(label_one_hot_encoding[0], opt.label_nc)
    out = np.concatenate((label, fake_image), axis = 0)
    cv2.imwrite("%s/test_img_%d.jpg" % (opt.results_dir, i), out)
