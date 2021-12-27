import os
import cv2
from data.base_method.image_option import loaded_label2ndarray, loaded_image2ndarray, np_transform
from data.base_method.image_folder import make_dataset
from PIL import Image
import numpy as np
import random

class Dataset_Help(object):
    def __init__(self, opt):
        super(Dataset_Help, self).__init__()
        self.opt = opt
        self.root = opt.dataroot

        # label
        if opt.label_nc <1:
            raise ('Need label_nc')
        self.dir_label = os.path.join(opt.dataroot, opt.phase+'_label')
        self.dir_label = sorted(make_dataset(self.dir_label))

        # real_image
        if opt.phase == 'train':
            self.dir_real_image = os.path.join(opt.dataroot, opt.phase+'_img')
            self.dir_real_image = sorted(make_dataset(self.dir_real_image))

        # instance maps
        if not opt.no_instance:
            self.instant_paths = os.path.join(opt.dataroot, opt.phase+'_inst')
            self.instant_paths = sorted(make_dataset(self.instant_paths))
        if opt.phase == 'train':
            self.shuffle2()


    def __getitem__(self, item):
        # label
        label_path = self.dir_label[item]
        label = cv2.imread(label_path, 0)

        label_2ndarray = loaded_label2ndarray(label, self.opt)

        # real image
        if self.opt.phase =='train':
            image_path = self.dir_real_image[item]
            image = cv2.imread(image_path)
            # cv2.imshow('1', image)
            # cv2.waitKey()
            image_2ndarray = loaded_image2ndarray(image, self.opt)
        # cv2.imshow('2', image_2ndarray[0].transpose(1, 2, 0))
        # cv2.waitKey()
        # inst maps
        if not self.opt.no_instance:
            inst_path = self.instant_paths[item]
            inst = Image.open(inst_path)
            inst_nd = np_transform(inst, self.opt, method=Image.NEAREST, normalize=False)
            inst_nd = np.expand_dims(inst_nd, axis=0)
            inst_nd = np.expand_dims(inst_nd, axis=0)

        if self.opt.phase == 'train':
            input_dict = {'label': label_2ndarray, 'real_image':image_2ndarray, 'instance':inst_nd}
        else:
            input_dict = {'label': label_2ndarray, 'instance': inst_nd}
        return input_dict


    def __len__(self):
        return len(self.dir_label)

    def lenOfIter_perBatch(self):
        return int(len(self.dir_label)) // int(self.opt.batch_size)

    def shuffle2(self):
        # use same random seed to shuffle list

        # every time use random seed in case same shuffle in different epoch
        ra = random.random()
        random.seed(ra)
        random.shuffle(self.dir_label)
        random.seed(ra)
        random.shuffle(self.dir_real_image)
        random.seed(ra)
        random.shuffle(self.instant_paths)