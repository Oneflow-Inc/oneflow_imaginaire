from PIL import Image
import numpy as np
import cv2

def loaded_image2ndarray(image, opt, method=cv2.INTER_CUBIC):
    h, w, c = image.shape
    # w = opt.load_size
    # h = int(opt.load_size * h/w)

    h, w = opt.my_size_h, opt.my_size_w
    image = cv2.resize(image, (w, h), interpolation=method)
    if opt.flip:
        image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = ((image.astype(np.float32) / 255.0) -0.5) /0.5 # [-1, 1]
    image = np.expand_dims(image, axis=0)
    return np.ascontiguousarray(image, 'float32')

def loaded_label2ndarray(image, opt, method=cv2.INTER_NEAREST):
    h, w = image.shape
    # w = opt.load_size
    # h = int(opt.load_size * h / w)
    h, w = opt.my_size_h, opt.my_size_w
    image = cv2.resize(image, (w, h), interpolation=method)
    if opt.flip:
        image = cv2.flip(image, 1)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return np.ascontiguousarray(image, 'float32')

def np_transform(input_nd, opt, method=Image.BICUBIC, normalize=True):
    out_nd = input_nd
    if 'resize' in opt.resize_or_crop:
        out_nd = out_nd.resize((opt.my_size_w, opt.my_size_h), method)
    elif 'scale_width' in opt.resize_or_crop:
        out_nd = __scale_width(out_nd, opt.my_size, method)

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        out_nd = __make_power_2(out_nd, base, method)

    if opt.flip:
        out_nd = __flip(out_nd, opt.flip)

    out_nd = np.array(out_nd)
    if normalize:
        out_nd = ((out_nd.astype(np.float) / 255.0) - 0.5) / 0.5

    return np.ascontiguousarray(out_nd.astype(np.float), 'float32')

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    # w = target_width
    # h = int(target_width * oh / ow)
    h, w = target_width, target_width
    return img.resize((w, h), method)

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img