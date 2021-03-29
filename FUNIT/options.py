import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataset_dir', type=str, help='')
        self.parser.add_argument('--checkpoints_dir', type=str, help='')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='')
        self.parser.add_argument('--seed', type=int, default=1, help='')

        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--num_image_channels', type=int, default=3, help='# of image channels')
        self.parser.add_argument('--image_size', type=int, default=256, help='input image size')
        self.parser.add_argument('--label_nc', type=int, default=119, help='# of label channels')
        self.parser.add_argument('--random_scale_limit', type=float, default=0.1, help='')

        # for generator
        self.parser.add_argument('--gen_num_filters', type=int, default=64, help='')
        self.parser.add_argument('--gen_num_filters_mlp', type=int, default=256, help='')
        self.parser.add_argument('--gen_style_dims', type=int, default=64, help='')
        self.parser.add_argument('--gen_num_res_blocks', type=int, default=2, help='')
        self.parser.add_argument('--gen_num_mlp_blocks', type=int, default=3, help='')
        self.parser.add_argument('--gen_num_downsamples_content', type=int, default=4, help='')
        self.parser.add_argument('--gen_num_downsamples_style', type=int, default=5, help='')

        # for discriminator
        self.parser.add_argument('--dis_num_filters', type=int, default=64, help='')
        self.parser.add_argument('--dis_max_num_filters', type=int, default=1024, help='')
        self.parser.add_argument('--dis_num_layers', type=int, default=6, help='')

        # for loss
        self.parser.add_argument('--gan_weight', type=float, default=1., help='weight of gan loss')
        self.parser.add_argument('--image_recon_weight', type=float, default=0.1, help='weight of image reconstruction loss')
        self.parser.add_argument('--feature_matching_weight', type=float, default=1., help='weight of feature matching loss')

        # for optimizer
        self.parser.add_argument('--epoch', type=int, default=10000, help='# of epoch')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
