from models.pix2pix_model import Pix2PixModel
import oneflow as flow
import oneflow.typing as tp
import numpy as np
from options import BaseOptions
from pre_process import preprocess_input
import cv2
from data.base_method.what_name import Dataset_Help
import util.util as util

opt = BaseOptions().parse()
opt.phase = 'test'
device_type = 'gpu' if opt.gpu_nums > 0 else 'cpu'
# device_type = 'cpu'
if device_type == 'gpu':
    flow.config.gpu_device_num(opt.gpu_nums)

flow.env.init()
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
# func_config.default_logical_view(flow.scope.consistent_view())
# func_config.default_placement_scope(flow.scope.placement("gpu", "0:2"))


batch = opt.batch_size
label_class_num = opt.label_nc
if not opt.no_instance:
    label_class_num+=1
image_channel = opt.input_nc
height, width = opt.my_size_h, opt.my_size_w
pix2pix = Pix2PixModel(opt)
dataset = Dataset_Help(opt)

@flow.global_function('predict', func_config)
def InferenceG(
    input_semantics_32: tp.Numpy.Placeholder((opt.batch_size, 37, height//32, width//32), dtype=flow.float),
    input_semantics_16: tp.Numpy.Placeholder((opt.batch_size, 37, height//16, width//16), dtype=flow.float),
    input_semantics_8: tp.Numpy.Placeholder((opt.batch_size, 37, height//8, width//8), dtype=flow.float),
    input_semantics_4: tp.Numpy.Placeholder((opt.batch_size, 37, height//4, width//4), dtype=flow.float),
    input_semantics_2: tp.Numpy.Placeholder((opt.batch_size, 37, height//2, width//2), dtype=flow.float),
    input_semantics_1: tp.Numpy.Placeholder((opt.batch_size, 37, height, width), dtype=flow.float),
):
    # with flow.scope.placement('gpu', '0:2'):
    fake_image, kld_loss = pix2pix.generate_fake(input_semantics_32, input_semantics_16, input_semantics_8, input_semantics_4, input_semantics_2, input_semantics_1, None, opt, trainable=False)
    return fake_image

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
            is_32[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (w // 32, h // 32))
            is_16[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (w // 16, h // 16))
            is_8[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (w // 8, h // 8))
            is_4[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (w // 4, h // 4))
            is_2[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (w // 2, h // 2))
            is_1[b][c_] = cv2.resize(input_semantics[b, c_, :, :].reshape((h, w, 1)), (w // 1, h // 1))
    return is_32, is_16, is_8, is_4, is_2, is_1

if opt.pre_G_D!='':
    flow.load_variables(flow.checkpoint.get(opt.pre_G_D))
    print('Load checkpoint G and D success')

for i in range(dataset.lenOfIter_perBatch()):
    data_dict = dataset[i]
    # image = data_dict['real_image']
    label = data_dict['label']
    instance = data_dict['instance']
    if opt.batch_size != 1:
        for b in range(1, opt.batch_size):
            data_dict = dataset[i+b]
            # image_ = data_dict['real_image']
            label_ = data_dict['label']
            instance_ = data_dict['instance']
            # image = np.concatenate((image, image_), axis=0)
            label = np.concatenate((label, label_), axis=0)
            instance = np.concatenate((instance, instance_), axis=0)
        i = i+b
    # data = {'label': label, 'image': image, 'instance': instance}
    data = {'label': label, 'instance': instance}
    input_semantics, real_image = preprocess_input(data, opt)

    is_32, is_16, is_8, is_4, is_2, is_1 = out_scale_semantic(input_semantics)

    fake_image = InferenceG(is_32, is_16, is_8, is_4, is_2, is_1).get()
    fake = util.tensor2im(fake_image.numpy()[0])
    label = util.onehot2label(is_1[0], opt.label_nc)
    out = np.concatenate((label, fake), axis=0)
    cv2.imwrite('inference'+str(i)+'_.jpg', out)

