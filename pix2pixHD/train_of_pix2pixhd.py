import os
import numpy as np
import argparse
import cv2
import random

import oneflow as flow
import oneflow.typing as tp

import networks
import util.util as util
import util.image_pool as image_pool

from vgg16_model import VGGLoss

from data.aligned_dataset import AlignedDataset
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

# load dataset
dataset = AlignedDataset()
print("dataset [%s] was created" % (dataset.name()))
dataset.initialize(opt)

batch, channel, height, width = dataset[0]["image"].shape

inst_map_channel = 1
label_class_num = opt.label_nc + inst_map_channel
image_channel = opt.input_nc

@flow.global_function("train", func_config)
def TrainDiscriminator(
    real_image: tp.Numpy.Placeholder((opt.batchSize, image_channel + label_class_num, height, width), dtype = flow.float32),
    fake_image_pool: tp.Numpy.Placeholder((opt.batchSize, image_channel + label_class_num, height, width), dtype = flow.float32)):
    # Calculate GAN loss for discriminator
    # Fake Detection and Loss
    pred_fake_pool = networks.MultiscaleDiscriminator(fake_image_pool, ndf=opt.ndf, n_layers=opt.n_layers_D, norm_type=opt.norm,
                                                    use_sigmoid=False, num_D=opt.num_D, trainable=True, reuse=True)
    
    loss_D_fake = networks.GANLoss(pred_fake_pool, False)

    # Real Detection and Loss
    pred_real = networks.MultiscaleDiscriminator(real_image, ndf=opt.ndf, n_layers=opt.n_layers_D, norm_type=opt.norm,
                                                    use_sigmoid=False, num_D=opt.num_D, trainable=True, reuse=True)
    
    loss_D_real = networks.GANLoss(pred_real, True)

    # Combined loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5

    flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [opt.lr]), beta1=opt.beta1).minimize(loss_D)

    return loss_D


@flow.global_function("train", func_config)
def TrainGenerators(
    label: tp.Numpy.Placeholder((opt.batchSize, label_class_num, height, width), dtype = flow.float32),
    image: tp.Numpy.Placeholder((opt.batchSize, image_channel, height, width), dtype = flow.float32),):
    fake_image = networks.define_G(label, opt.output_nc, opt.ngf, opt.netG,
                                n_downsample_global=opt.n_downsample_global, n_blocks_global=opt.n_blocks_global,
                                n_blocks_local=opt.n_blocks_local, norm_type=opt.norm, trainable=True, reuse=True)

    # GAN loss (Fake Passability Loss)
    fake_image_concat_label = flow.concat([label, fake_image], axis=1)
    pred_fake = networks.MultiscaleDiscriminator(fake_image_concat_label, ndf=opt.ndf, n_layers=opt.n_layers_D, norm_type=opt.norm,
                                                use_sigmoid=False, num_D=opt.num_D, trainable=False, reuse=True)
    loss_G_GAN = networks.GANLoss(pred_fake, True)

    real_image_concat_label = flow.concat([label, image], axis=1)

    pred_real = networks.MultiscaleDiscriminator(real_image_concat_label, ndf=opt.ndf, n_layers=opt.n_layers_D, norm_type=opt.norm,
                                                use_sigmoid=False, num_D=opt.num_D, trainable=False, reuse=True)
    
    # GAN feature matching loss
    loss_G_GAN_Feat = 0
    feat_weights = 4.0 / (opt.n_layers_D + 1)
    D_weights = 1.0 / opt.num_D
    weight = D_weights * feat_weights * opt.lambda_feat
    for i in range(opt.num_D):
        for j in range(len(pred_fake[i])-1):
            loss_G_GAN_Feat = flow.nn.L1Loss(pred_fake[i][j], pred_real[i][j]) + loss_G_GAN_Feat       

    # combined loss and calculate gradients
    loss_G = loss_G_GAN + loss_G_GAN_Feat * weight

    if not opt.no_vgg_loss:
        loss_G = loss_G + VGGLoss(fake_image, image) * opt.lambda_feat

    flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [opt.lr]), beta1=opt.beta1).minimize(loss_G)

    return fake_image, fake_image_concat_label, real_image_concat_label, loss_G


fake_pool = image_pool.ImagePool(opt.pool_size)
epoch = 1000
dataset_len = len(dataset)


if opt.load_pretrain != "":
    flow.load_variables(flow.checkpoint.get(opt.load_pretrain))

for e in range(epoch):
    e = e + 32
    for i in range(dataset_len):
        data_dict = dataset[i]

        label_one_hot_encoding = np.zeros((batch, opt.label_nc, height, width), dtype=np.float)
        util.scatter(label_one_hot_encoding, 1, data_dict['label'].astype(np.int32), 1)
        edge_nd = util.get_inst_map_edge(data_dict["inst"].astype(np.int32))
        label_nd = np.concatenate((label_one_hot_encoding, edge_nd.astype(np.float)), axis = 1)
    
        fake_image, fake_image_concat_label, real_image_concat_label, loss_G = TrainGenerators(label_nd, data_dict["image"]).get()
        
        fake_image_pool = fake_pool.query(fake_image_concat_label.numpy())
        loss_D = TrainDiscriminator(real_image_concat_label.numpy(), fake_image_pool).get()
        print("epoch %d, iter %d, GL_oss: " % (e, i), loss_G.numpy(), "D_Loss: ", loss_D.numpy())

        if i % 5 == 0:
            real = util.tensor2im(data_dict['image'][0])
            fake = util.tensor2im(fake_image.numpy()[0])
            label = util.onehot2label(label_one_hot_encoding[0], opt.label_nc)
            out = np.concatenate((label, fake, real), axis = 0)
            edge_img = np.squeeze(edge_nd, axis=0)
            edge_img = np.transpose(edge_img, (1, 2, 0))
            edge_img *= 255
            cv2.imwrite("train_tmp_instance.jpg", edge_img)
            cv2.imwrite("train_tmp_result.jpg", out)

        if i % 300 == 0:
            flow.checkpoint.save("%s/opech_%d_iter_%i_Gloss_%f_Dloss_%f" % (opt.checkpoints_dir, e, i, loss_G.numpy()[0], loss_D.numpy()[0]))

#     label_one_hot_encoding = np.zeros((batch, opt.label_nc, height, width))
#     util.scatter(label_one_hot_encoding, 1, data_dict['label'].astype(np.int32), 1)

    # print(label_one_hot_encoding.shape)

    



#     edge_nd = util.get_inst_map_edge(data_dict["inst"])

#     print(edge_nd.shape)

#     # edge_img = np.transpose(edge_nd[0], (1, 2, 0)) * 255





# concat one-hot label_nd and edge inst_nd

# fake_pool = image_pool.ImagePool(opt.pool_size)

# label_nd = np.zeros((opt.batchSize, label_class_num, height, width))
# inst_nd = np.zeros((opt.batchSize, inst_map_channel, height, width))
# image_nd = np.zeros((opt.batchSize, image_channel, height, width))

# for i in range(1000):
#     fake_image, fake_image_concat_label, real_image_concat_label, loss_G = TrainGenerators(label_nd, image_nd).get()
#     fake_image_pool = fake_pool.query(fake_image_concat_label.numpy())
#     # loss_D = TrainDiscriminator(real_image_concat_label.numpy(), fake_image_pool).get()
#     # print(fake_image_pool.shape, loss_G.numpy(), loss_D.numpy())
#     flow.checkpoint.save("./checkpoint")


   
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






