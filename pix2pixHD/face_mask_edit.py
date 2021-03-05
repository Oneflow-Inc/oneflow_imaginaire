import cv2
import numpy as np
import os
import sys
import time
from PIL import Image
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

import oneflow as flow
import oneflow.typing as tp

from data.base_dataset import load_label2ndarray
import models.networks as networks
from options.test_options import TestOptions
import util.util as util
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
from ui_util.config import Config


color_list = [QColor(0, 0, 0), QColor(204, 0, 0), QColor(76, 153, 0), QColor(204, 204, 0), QColor(51, 51, 255), QColor(204, 0, 204), QColor(0, 255, 255), QColor(51, 255, 255), QColor(102, 51, 0), QColor(255, 0, 0), QColor(102, 204, 0), QColor(255, 255, 0), QColor(0, 0, 153), QColor(0, 0, 204), QColor(255, 51, 153), QColor(0, 204, 204), QColor(0, 51, 0), QColor(255, 153, 51), QColor(0, 204, 0)]

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
opt = TestOptions().parse()
opt.batchSize = 1
opt.gpu_nums = 1
opt.label_nc = 19

flow.config.gpu_device_num(opt.gpu_nums)
flow.env.init()
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_logical_view(flow.scope.consistent_view())
func_config.default_placement_scope(flow.scope.placement("gpu", "0:0"))

@flow.global_function("predict", func_config)
def TrainGenerators(
    label: tp.Numpy.Placeholder((opt.batchSize, opt.label_nc, opt.loadSize, opt.loadSize), dtype = flow.float32),):
    fake_image = networks.define_G(label, opt.output_nc, opt.ngf, opt.netG,
                                n_downsample_global=opt.n_downsample_global, n_blocks_global=opt.n_blocks_global,
                                n_blocks_local=opt.n_blocks_local, norm_type=opt.norm, trainable=False, reuse=True)
    return fake_image

assert opt.load_pretrain != ""
flow.load_variables(flow.checkpoint.get(opt.load_pretrain))

class Ex(QWidget, Ui_Form):
    def __init__(self, opt):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()
        self.opt = opt

        self.output_img = None

        self.mat_img = None

        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.img = None

        self.mouse_clicked = False
        self.scene = GraphicsScene(self.mode, self.size)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.result_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

    def open_mask(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:    
            mat_img = cv2.imread(fileName)
            self.mask = mat_img.copy()
            self.mask_m = mat_img       
            mat_img = mat_img.copy()
            image = QImage(mat_img, 512, 512, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return    

            for i in range(512):
                for j in range(512):
                    r, g, b, a = image.pixelColor(i, j).getRgb()
                    image.setPixel(i, j, color_list[r].rgb()) 
           
            pixmap = QPixmap()
            pixmap.convertFromImage(image)  
            self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items() 
            self.scene.addPixmap(self.image)

    def bg_mode(self):
        self.scene.mode = 0

    def skin_mode(self):
        self.scene.mode = 1

    def nose_mode(self):
        self.scene.mode = 2

    def eye_g_mode(self):
        self.scene.mode = 3

    def l_eye_mode(self):
        self.scene.mode = 4

    def r_eye_mode(self):
        self.scene.mode = 5

    def l_brow_mode(self):
        self.scene.mode = 6

    def r_brow_mode(self):
        self.scene.mode = 7

    def l_ear_mode(self):
        self.scene.mode = 8

    def r_ear_mode(self):
        self.scene.mode = 9

    def mouth_mode(self):
        self.scene.mode = 10

    def u_lip_mode(self):
        self.scene.mode = 11

    def l_lip_mode(self):
        self.scene.mode = 12

    def hair_mode(self):
        self.scene.mode = 13

    def hat_mode(self):
        self.scene.mode = 14

    def ear_r_mode(self):
        self.scene.mode = 15

    def neck_l_mode(self):
        self.scene.mode = 16

    def neck_mode(self):
        self.scene.mode = 17

    def cloth_mode(self):
        self.scene.mode = 18
    
    def increase(self):
        if self.scene.size < 15:
            self.scene.size += 1
    
    def decrease(self):
        if self.scene.size > 1:
            self.scene.size -= 1 

    def edit(self):
        print("fake")
        for i in range(19):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)

        mask_m = self.mask_m.copy()
        mask_m = cv2.cvtColor(mask_m, cv2.COLOR_BGR2GRAY)
        mask_m = load_label2ndarray(mask_m, self.opt, False)

        start_t = time.time()
        
        label_one_hot_encoding = np.zeros((opt.batchSize, opt.label_nc, opt.loadSize, opt.loadSize), dtype=np.float)
        util.scatter(label_one_hot_encoding, 1, mask_m.astype(np.int32), 1)
        label_nd = label_one_hot_encoding
        
        generated = TrainGenerators(label_nd).get()
        
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))

        result = util.tensor2im(generated.numpy()[0])
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        qim = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)

        if len(self.result_scene.items()) == 0:
            self.result_scene.addPixmap(QPixmap.fromImage(qim))
        elif len(self.result_scene.items())>0: 
            self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(QPixmap.fromImage(qim))

    def make_mask(self, mask, pts, sizes, color):
        if len(pts)>0:
            for idx, pt in enumerate(pts):
                cv2.line(mask,pt['prev'],pt['curr'],(color,color,color),sizes[idx])
        return mask

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                    QDir.currentPath())
            cv2.imwrite(fileName+'.jpg',self.output_img)

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.mask_m = self.mask.copy()
    
        self.scene.reset_items()
        self.scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)


app = QApplication(sys.argv)
ex = Ex(opt)
sys.exit(app.exec_())
