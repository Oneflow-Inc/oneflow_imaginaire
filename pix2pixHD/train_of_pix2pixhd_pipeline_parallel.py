import os
import numpy as np
import argparse
import cv2
import random

import oneflow as flow
import oneflow.typing as tp

import networks_pipeline_parallel as networks
import image_pool

from options.train_options import TrainOptions

opt = TrainOptions().parse()

device_type = "gpu" if opt.gpu_nums > 0 else "cpu"
if device_type == "gpu":
    flow.config.gpu_device_num(opt.gpu_nums)

flow.env.init()
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_logical_view(flow.scope.consistent_view())
func_config.default_placement_scope(flow.scope.placement("gpu", "0:0"))

# flow.config.enable_debug_mode(True)


# make input shape for cityscape
height = 1024
width = 2048
if 'crop' in opt.resize_or_crop:
    height = opt.fineSize
    width = opt.fineSize
elif opt.resize_or_crop == 'scale_width':
    height = int(opt.loadSize / 2)
    width = opt.loadSize
cityscape_label_class_num = 36
cityscape_inst_map_channel = 1
cityscape_image_channel = 3

# (TODO:Liangdepeng) load datasets

@flow.global_function("train", func_config)
def TrainDiscriminator(
    real_image: tp.Numpy.Placeholder((opt.batchSize, cityscape_image_channel + cityscape_label_class_num, height, width), dtype = flow.float32),
    fake_image_pool: tp.Numpy.Placeholder((opt.batchSize, cityscape_image_channel + cityscape_label_class_num, height, width), dtype = flow.float32)):
    with flow.scope.placement("gpu", "0:0"): 
        # Calculate GAN loss for discriminator
        # Fake Detection and Loss
        pred_fake_pool = networks.MultiscaleDiscriminator(fake_image_pool, ndf=opt.ndf, n_layers=opt.n_layers_D, norm_type=opt.norm,
                                                        use_sigmoid=opt.no_lsgan, num_D=opt.num_D, trainable=True, reuse=True)
        
        loss_D_fake = networks.GANLoss(pred_fake_pool, False)

    with flow.scope.placement("gpu", "0:0"):
        # Real Detection and Loss
        pred_real = networks.MultiscaleDiscriminator(real_image, ndf=opt.ndf, n_layers=opt.n_layers_D, norm_type=opt.norm,
                                                        use_sigmoid=opt.no_lsgan, num_D=opt.num_D, trainable=True, reuse=True)
        
        loss_D_real = networks.GANLoss(pred_real, True)

        # Combined loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [opt.lr]), beta1=opt.beta1, beta2=0.999).minimize(loss_D)

    return loss_D


@flow.global_function("train", func_config)
def TrainGenerators(
    label: tp.Numpy.Placeholder((opt.batchSize, cityscape_label_class_num, height, width), dtype = flow.float32),
    image: tp.Numpy.Placeholder((opt.batchSize, cityscape_image_channel, height, width), dtype = flow.float32),):
    

    # with flow.scope.placement("gpu", "0:0"): 
    fake_image = networks.define_G(label, opt.output_nc, opt.ngf, opt.netG,
                                n_downsample_global=opt.n_downsample_global, n_blocks_global=opt.n_blocks_global,
                                n_blocks_local=opt.n_blocks_local, norm_type=opt.norm, trainable=True, reuse=True)

    with flow.scope.placement("gpu", "0:0"): 
        # GAN loss (Fake Passability Loss)
        fake_image_concat_label = flow.concat([label, fake_image], axis=1)
        pred_fake = networks.MultiscaleDiscriminator(fake_image_concat_label, ndf=opt.ndf, n_layers=opt.n_layers_D, norm_type=opt.norm,
                                                    use_sigmoid=opt.no_lsgan, num_D=opt.num_D, trainable=False, reuse=True)
        loss_G_GAN = networks.GANLoss(pred_fake, True)

    with flow.scope.placement("gpu", "0:0"):
        real_image_concat_label = flow.concat([label, image], axis=1)
        pred_real = networks.MultiscaleDiscriminator(real_image_concat_label, ndf=opt.ndf, n_layers=opt.n_layers_D, norm_type=opt.norm,
                                                    use_sigmoid=opt.no_lsgan, num_D=opt.num_D, trainable=False, reuse=True)   


    with flow.scope.placement("gpu", "0:0"): 
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (opt.n_layers_D + 1)
        D_weights = 1.0 / opt.num_D
        for i in range(opt.num_D):
            for j in range(len(pred_fake[i])-1):
                loss_G_GAN_Feat += D_weights * feat_weights * opt.lambda_feat * flow.nn.L1Loss(pred_fake[i][j], pred_real[i][j])                 

        # combined loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_GAN_Feat
        flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [opt.lr]), beta1=opt.beta1, beta2=0.999).minimize(loss_G)

    return fake_image, fake_image_concat_label, real_image_concat_label, loss_G

label_nd = np.zeros((opt.batchSize, cityscape_label_class_num, height, width))
inst_nd = np.zeros((opt.batchSize, cityscape_inst_map_channel, height, width))
image_nd = np.zeros((opt.batchSize, cityscape_image_channel, height, width))

# concat one-hot label_nd and edge inst_nd

fake_pool = image_pool.ImagePool(opt.pool_size)

for i in range(1000):
    fake_image, fake_image_concat_label, real_image_concat_label, loss_G = TrainGenerators(label_nd, image_nd).get()
    fake_image_pool = fake_pool.query(fake_image_concat_label.numpy())
    loss_D = TrainDiscriminator(real_image_concat_label.numpy(), fake_image_pool).get()
    print(fake_image_pool.shape, loss_G.numpy(), loss_D.numpy())
    # flow.checkpoint.save("./checkpoint")


   
# result = fake_pool.query(test)
# print(len(fake_pool.images))
# for image in fake_pool.images:
#     print(image.shape)
# print(result.shape)

# result = fake_pool.query(test)
# print("round2")
# print(len(fake_pool.images))
# for image in fake_pool.images:
#     print(image.shape)
# print(result.shape)






