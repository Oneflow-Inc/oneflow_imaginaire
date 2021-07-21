from models.pix2pix_model import Pix2PixModel
import oneflow as flow
import oneflow.typing as tp
import numpy as np
from options import BaseOptions
from pre_process import preprocess_input
import cv2
from data.base_method.what_name import Dataset_Help
from util.spectral_norm import spectral_norm

opt = BaseOptions().parse()

device_type = 'gpu' if opt.gpu_nums > 0 else 'cpu'
if device_type == 'gpu':
    flow.config.gpu_device_num(opt.gpu_nums)

flow.env.init()
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
# func_config.default_logical_view(flow.scope.consistent_view)
func_config.default_placement_scope(flow.scope.placement('gpu', '0:0'))

fake_dataset = True
if fake_dataset:
    print('Use fake dataset')
else:
    raise ('Have not load dataset')

batch = opt.batch_size
label_class_num = opt.label_nc
if not opt.no_instance:
    label_class_num+=1
image_channel = opt.input_nc
height, width = opt.my_size_h, opt.my_size_w
pix2pix = Pix2PixModel(opt)
dataset = Dataset_Help(opt)

@flow.global_function('train', func_config)
def TrainD(
    input_semantics_32: tp.Numpy.Placeholder((opt.batch_size, 37, height//32, width//32), dtype=flow.float),
    input_semantics_16: tp.Numpy.Placeholder((opt.batch_size, 37, height//16, width//16), dtype=flow.float),
    input_semantics_8: tp.Numpy.Placeholder((opt.batch_size, 37, height//8, width//8), dtype=flow.float),
    input_semantics_4: tp.Numpy.Placeholder((opt.batch_size, 37, height//4, width//4), dtype=flow.float),
    input_semantics_2: tp.Numpy.Placeholder((opt.batch_size, 37, height//2, width//2), dtype=flow.float),
    input_semantics_1: tp.Numpy.Placeholder((opt.batch_size, 37, height, width), dtype=flow.float),
    real_image: tp.Numpy.Placeholder((opt.batch_size, 3, height, width),
                                         dtype=flow.float32),

):
    d_losses = pix2pix.compute_D_loss(input_semantics_32, input_semantics_16, input_semantics_8,
                                                 input_semantics_4, input_semantics_2, input_semantics_1, real_image)
    # loss = 0
    # for k in d_losses.keys():
    #     loss += d_losses[k]
    loss = sum(d_losses.values())
    flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [opt.lr]), beta1=opt.beat1).minimize(loss)
    return d_losses


@flow.global_function('train', func_config)
def TrainG(
    input_semantics_32: tp.Numpy.Placeholder((opt.batch_size, 37, height//32, width//32), dtype=flow.float),
    input_semantics_16: tp.Numpy.Placeholder((opt.batch_size, 37, height//16, width//16), dtype=flow.float),
    input_semantics_8: tp.Numpy.Placeholder((opt.batch_size, 37, height//8, width//8), dtype=flow.float),
    input_semantics_4: tp.Numpy.Placeholder((opt.batch_size, 37, height//4, width//4), dtype=flow.float),
    input_semantics_2: tp.Numpy.Placeholder((opt.batch_size, 37, height//2, width//2), dtype=flow.float),
    input_semantics_1: tp.Numpy.Placeholder((opt.batch_size, 37, height, width), dtype=flow.float),
    real_image: tp.Numpy.Placeholder((opt.batch_size, image_channel, height, width), dtype=flow.float32)
):
    g_losses, fake_image = pix2pix.compute_G_loss(input_semantics_32, input_semantics_16, input_semantics_8,
                                                 input_semantics_4, input_semantics_2, input_semantics_1, real_image, opt, trainable=True)
    loss = sum(g_losses.values())
    flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [opt.lr]), beta1=opt.beat1).minimize(loss)
    return g_losses, fake_image


# 在SAPDE官方代码里面调试
# label.shape = [1, 1, 256, 256]
# instance.shape = [1, 1, 256, 256]
# image.shape = [1, 3, 256, 256]
# input_semantics.shape = [1, 183, 256, 256]

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:,:,0]
    image_numpy = image_numpy.astype(imtype)
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    return image_numpy

def out_scale_semantic(input_semantics):
    bs, c, h, w = input_semantics.shape
    is_32 = np.zeros((bs, c, h // 32, w // 32))
    is_16 = np.zeros((bs, c, h // 16, w // 16))
    is_8 = np.zeros((bs, c, h // 8, w // 8))
    is_4 = np.zeros((bs, c, h // 4, w // 4))
    is_2 = np.zeros((bs, c, h // 2, w // 2))
    is_1 = np.zeros((bs, c, h // 1, w // 1))
    for b in range(bs):  # 遍历 batchsize
        for c_ in range(c):  # 遍历 通道
            is_32[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (h // 32, w // 32))
            is_16[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (h // 16, w // 16))
            is_8[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (h // 8, w // 8))
            is_4[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (h // 4, w // 4))
            is_2[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (h // 2, w // 2))
            is_1[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (h // 1, w // 1))
    # cv2.imshow('0', input_semantics[0][22].astype(np.float32))
    # cv2.waitKey()
    # cv2.imshow('1', is_32[0][22].astype(np.float32))
    # cv2.waitKey()
    # cv2.imshow('2', is_16[0][22].astype(np.float32))
    # cv2.waitKey()
    # cv2.imshow('3', is_8[0][22].astype(np.float32))
    # cv2.waitKey()
    # cv2.imshow('4', is_4[0][22].astype(np.float32))
    # cv2.waitKey()
    # cv2.imshow('5', is_2[0][22].astype(np.float32))
    # cv2.waitKey()
    # cv2.imshow('6', is_1[0][22].astype(np.float32))
    # cv2.waitKey()
    return is_32, is_16, is_8, is_4, is_2, is_1


for epoch in range(opt.epochs):
    for i in range(len(dataset)):
        data_dict = dataset[i]
        image = data_dict['real_image']
        label = data_dict['label']
        instance = data_dict['instance']
        if opt.batch_size != 1:
            for b in range(1, opt.batch_size):
                data_dict = dataset[i+b]
                image_ = data_dict['real_image']
                label_ = data_dict['label']
                instance_ = data_dict['instance']
                image = np.concatenate((image, image_), axis=0)
                label = np.concatenate((label, label_), axis=0)
                instance = np.concatenate((instance, instance_), axis=0)
            i = i+b
        data = {'label': label, 'image': image, 'instance': instance}
        # cv2.imshow('1', data['label'][0].reshape(256, 256, 1).astype(np.int8))
        # cv2.waitKey()
        # cv2.imshow('2', data['image'][0].reshape(256, 256, 3).astype(np.float))
        # cv2.waitKey()
        # cv2.imshow('3', data['instance'][0].reshape(256, 256, 1).astype(np.int8))
        # cv2.waitKey()
        input_semantics, real_image = preprocess_input(data, opt)

        is_32, is_16, is_8, is_4, is_2, is_1 = out_scale_semantic(input_semantics)

        g_loss, fake_image = TrainG(is_32, is_16, is_8, is_4, is_2, is_1, real_image).get()
        b = TrainD(is_32, is_16, is_8, is_4, is_2, is_1, real_image).get()
        print('G:', end='')
        for k in g_loss.keys():
            print(k, g_loss[k].numpy(), end='')
        print('B:', end='')
        for k in b.keys():
            print(k, b[k].numpy(), end='')
        print(' ')
        weight_dict = flow.get_all_variables()
        weight_copy = {}
        for k in weight_dict.keys():
            weight_copy[k] = np.zeros(weight_dict[k].numpy().shape)

        for k in weight_dict.keys():
            weight_temp = weight_dict[k].numpy()
            weight_copy[k] = spectral_norm(weight_temp)
            # print(type(weight_dict[k]))
            # print(type(weight_copy[k]))

        flow.load_variables(weight_copy)


        if i % 10 ==0:
            real = tensor2im(data_dict['real_image'][0])
            fake = tensor2im(fake_image.numpy()[0])
            out = np.concatenate((fake, real), axis=0)
            cv2.imwrite('haha.jpg', out)

        if i % 300 ==0:
            flow.checkpoint.save('./checkpoints/haha')