import argparse
from util.save import saveDict_as_txt
import time

class BaseOptions():
    def __init__(self):
        super(BaseOptions, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experimental specifics
        self.parser.add_argument('--name', type=str, default='coco', help='name of the experiment')

        self.parser.add_argument('--gpu_nums', type=int, default=1, help='')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        self.parser.add_argument('--pre_vgg', type=str, default='./', help='used for perceptual loss')
        self.parser.add_argument('--pre_G_D', type=str, default='', help='')
        self.parser.add_argument('--sn', action='store_true', help='spade_norm')



        # input/output size
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--load_size', type=int, default=1024, help='')
        self.parser.add_argument('--crop_size', type=int, default=512, help='')
        self.parser.add_argument('--my_size_h', type=int, default=256, help='')
        self.parser.add_argument('--my_size_w', type=int, default=512, help='')
        self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        self.parser.add_argument('--contain_dontcare_label', action='store_true', default=True, help='if the label map contains dontcare label (dontcare=255)')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # input setting
        self.parser.add_argument('--dataroot', type=str, default='/home/shikaijie/dataset/cityscape/cityscapes_pix2pixHD', help='')
        self.parser.add_argument('--dataset_mode', type=str, default='coco', help='')
        self.parser.add_argument('--flip', action='store_true', help='')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize')

        # generator
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--z_dim', type=int, default=256)

        # instance-wise feature
        self.parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')


        self.parser.add_argument('--input_nc', type=int, default=3, help='')
        self.parser.add_argument('--epochs', type=int, default=100, help='')
        self.parser.add_argument('--height', type=int, default=256, help='')
        self.parser.add_argument('--width', type=int, default=512, help='')
        self.parser.add_argument('--lr_G', type=float, default=0.0001, help='')
        self.parser.add_argument('--lr_D', type=float, default=0.0004, help='')
        self.parser.add_argument('--beta1', type=float, default=0, help='')
        self.parser.add_argument('--beta2', type=float, default=0.9, help='')#issue 50
        self.parser.add_argument('--lambda_feta', type=float, default=10.0, help='')
        self.parser.add_argument('--lambda_vgg', type=float, default=10.0, help='')

        self.parser.add_argument('--num_image_channels', type=int, default=3, help='# of image channels')
        self.parser.add_argument('--image_size', type=int, default=256, help='input image size')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--isTrain', action='store_true', default=True, help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--up', type=str, default='nearest', help='nearest|bilinear|deconv|subpixel')


        # for genereator

        # for discriminator

        # for loss

        # for optimizer

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('----------------Options----------------')
        for k, v in sorted(args.items()):
            print(str(k)+':'+str(v))
        print('------------------END------------------')

        path = 'options->' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        saveDict_as_txt(path, args)

        return self.opt