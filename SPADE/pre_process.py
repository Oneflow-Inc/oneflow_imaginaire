import numpy as np
import cv2

def scatter(out_array, dim, index, value):  # a inplace
    expanded_index = [index if dim == i else np.arange(out_array.shape[i]).reshape(
        [-1 if i == j else 1 for j in range(out_array.ndim)]) for i in range(out_array.ndim)]
    # expanded_index = [index if dim == i else np.arange(out_array.shape[i]).reshape(
    #     [-1 if i == j else 1 for j in range(len(out_array.shape))]) for i in range(len(out_array.shape))]
    out_array[tuple(expanded_index)] = value

def get_edge(t):
    # print(t.shape)
    # edge = flow.zeros(t.shape, dtype=bytes)
    edge = np.zeros(shape=t.shape, dtype=np.bool)
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.astype(np.float)


def preprocess_input(data, opt):
    # data['label'] = data['label'].long()
    data['label'] = data['label'].astype(np.long)

    label_map = data['label']
    bs, _, h, w = label_map.shape
    nc = opt.label_nc + 1 if opt.contain_dontcare_label else opt.label_nc
    input_label = np.zeros(shape=[bs, nc, h, w], dtype=np.float32)
    # input_label 就是 input_semantics
    scatter(input_label, dim=1, index=label_map, value=1.0)
    # cv2.imshow('1', label_map[0][0].astype(np.int8))
    # cv2.waitKey()
    # cv2.imshow('2', input_label[0][22].astype(np.float32))
    # print(input_label[0][22])
    # cv2.waitKey()
    if not opt.no_instance:
        inst_map = data['instance']
        instance_edge_map = get_edge(inst_map)
        input_semantics = np.concatenate([input_label, instance_edge_map], 1)
        # cv2.imshow('2', instance_edge_map[0][0].astype(np.float32))
        # cv2.waitKey()
    if opt.phase == 'train':
        return input_semantics, data['image']
    else:
        return input_semantics, None

def pre_process_seg(segmap_raw):
    # size: [1, 2, 4, 8, 16, 32, 64, 128, 256]
    segmap_resize = {}
    bs, h, w, c = segmap_raw.shape # opencv 读入图片的格式

    segmap_resize['1'] = np.zeros(shape=(bs, h, w, c))
    # 256/2=128
    segmap_resize['1/2'] = np.zeros(shape=(bs, h//2, w//2, c))
    # 256/4=64
    segmap_resize['1/4'] = np.zeros(shape=(bs, h//4, w//4, c))
    # 256/8=32
    segmap_resize['1/8'] = np.zeros(shape=(bs, h//8, w//8, c))
    # 256/16=16
    segmap_resize['1/16'] = np.zeros(shape=(bs, h//16, w//16, c))
    # 256/32=8
    segmap_resize['1/32'] = np.zeros(shape=(bs, h//32, w//32, c))
    # 256/64=4
    segmap_resize['1/64'] = np.zeros(shape=(bs, h//64, w//64, c))

    for i in range(bs):
        segmap_resize['1'][i] = cv2.resize(segmap_raw[i], (h, w),).reshape((h, w, 1))
        segmap_resize['1/2'][i] = cv2.resize(segmap_raw[i], (h//2, w//2), interpolation=cv2.INTER_NEAREST).reshape((h//2, w//2, 1))
        segmap_resize['1/4'][i] = cv2.resize(segmap_raw[i], (h//4, w//4), interpolation=cv2.INTER_NEAREST).reshape((h//4, w//4, 1))
        segmap_resize['1/8'][i] = cv2.resize(segmap_raw[i], (h//8, w//8), interpolation=cv2.INTER_NEAREST).reshape((h//8, w//8, 1))
        segmap_resize['1/16'][i] = cv2.resize(segmap_raw[i], (h//16, w//16), interpolation=cv2.INTER_NEAREST).reshape((h//16, w//16, 1))
        segmap_resize['1/32'][i] = cv2.resize(segmap_raw[i], (h//32, w//32), interpolation=cv2.INTER_NEAREST).reshape((h//32, w//32, 1))
        segmap_resize['1/64'][i] = cv2.resize(segmap_raw[i], (h//64, w//64), interpolation=cv2.INTER_NEAREST).reshape((h//64, w//64, 1))

    return segmap_resize



if __name__ == '__main__':
    input = np.random.random((256, 256, 1))
    a = cv2.resize(input, (128, 128,), interpolation=cv2.INTER_NEAREST).reshape(128, 128, 1)
    print(a)
    seg_raw = np.random.random((2, 256, 256, 1))
    a = pre_process_seg(seg_raw)
    print(a)