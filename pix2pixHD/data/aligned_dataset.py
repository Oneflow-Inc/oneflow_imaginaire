import os.path
from data.base_dataset import get_params, normalize, np_transform, load_image2ndarray, load_label2ndarray
from data.image_folder import make_dataset
import numpy as np
import cv2
from PIL import Image
import random

class AlignedDataset(object):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):
        flip = random.random() > 0.5

        ### input A (label maps)
        A_path = self.A_paths[index]              
        A_nd = cv2.imread(A_path, 0)
        A_nd = load_label2ndarray(A_nd, self.opt, flip)

        inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B_nd = cv2.imread(B_path)
            B_nd = load_image2ndarray(B_nd, self.opt, flip)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            # pil
            inst = Image.open(inst_path)
            params = get_params(self.opt, inst.size)
            inst_nd = np_transform(inst, self.opt, flip, method=Image.NEAREST, normalize=False)
            inst_nd = np.expand_dims(inst_nd, axis=0)
            inst_nd = np.expand_dims(inst_nd, axis=0)
            # opencv
            # inst_nd = cv2.imread(inst_path, 0)
            # inst_nd = load_label2ndarray(inst_nd, self.opt)

        input_dict = {'label': A_nd, 'inst': inst_nd, 'image': B_nd, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
