from PIL import Image
import cv2
import numpy as np
import random

import cv2

def load_image2ndarray(im, opt, flip, method=cv2.INTER_CUBIC):
    oh, ow, oc = im.shape
    w = opt.loadSize
    h = int(opt.loadSize * oh / ow)
    im = cv2.resize(im, (w, h), interpolation = method)
    if flip:
        im = cv2.flip(im, 1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = ((im.astype(np.float32) / 255.0) - 0.5) / 0.5
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')

def load_label2ndarray(im, opt, flip, method=cv2.INTER_NEAREST):
    oh, ow = im.shape
    w = opt.loadSize
    h = int(opt.loadSize * oh / ow)
    im = cv2.resize(im, (w, h), interpolation = method)
    if flip:
        im = cv2.flip(im, 1)
    im = np.expand_dims(im, axis=0)
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')

def np_transform(input_nd, opt, flip, method=Image.BICUBIC, normalize=True):
    out_nd = input_nd
    if 'resize' in opt.resize_or_crop:
        out_nd = out_nd.resize((opt.loadSize, opt.loadSize), method)
    elif 'scale_width' in opt.resize_or_crop:
        out_nd = __scale_width(out_nd, opt.loadSize, method)

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        out_nd = __make_power_2(out_nd, base, method)

    if flip:
        out_nd = __flip(out_nd, flip)

    out_nd = np.array(out_nd)
    if normalize:
        out_nd = ((out_nd.astype(np.float) / 255.0) - 0.5) / 0.5

    return np.ascontiguousarray(out_nd.astype(np.float), 'float32')

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

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
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
