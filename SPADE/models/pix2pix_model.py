from .generator import generator
from .discriminator import multi_scale_discriminator
from .encoder import conv_encoder
import oneflow as flow
from .network import loss
import numpy as np


class Pix2PixModel():
    def __init__(self, opt):
        super(Pix2PixModel, self).__init__()
        self.opt = opt

        self.G, self.D, self.E = self.initialize_network(opt)

        if opt.isTrain:
            self.criterionGANLoss = loss.GAN_loss(gan_mode='hinge')
            self.criterionFeatureLoss = flow.nn.L1Loss
            if not opt.no_vgg_loss:
                self.criterionVggLoss = loss.VGG_loss()
            if opt.use_vae:
                self.KLDloss = loss.KLD_loss()

    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_G_loss(input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_D_loss(input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            # no grad:
            fake_image, _ = self.generate_fake(input_semantics, real_image, trainable=False)
            return fake_image
        else:
            raise ('MOOOO')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def initialize_network(self, opt):
        G = generator
        D = multi_scale_discriminator
        E = conv_encoder
        return G, D, E

    # generate onehot like pytorch scatter_ operation
    # default it's right (skj)
    def scatter(self, out_array, dim, index, value):  # a inplace
        expanded_index = [index if dim == i else np.arange(out_array.shape[i]).reshape(
            [-1 if i == j else 1 for j in range(out_array.ndim)]) for i in range(out_array.ndim)]
        # expanded_index = [index if dim == i else np.arange(out_array.shape[i]).reshape(
        #     [-1 if i == j else 1 for j in range(len(out_array.shape))]) for i in range(len(out_array.shape))]
        out_array[tuple(expanded_index)] = value

    def preprocess_input(self, data):
        # data['label'] = data['label'].long()
        # data['label'] = data['label'].astype(np.long)
        if self.use_gpu():
            print('data to GPU')

        label_map = data['label']
        bs, _, h, w = label_map.shape
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = np.zeros(shape=[bs, nc, h, w], dtype=np.float32)
        # input_label 就是 input_semantics
        self.scatter(input_label, dim=1, index=label_map, value=1.0)

        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edge(inst_map)
            input_semantics = flow.concat([input_label, instance_edge_map], 1)

        return input_semantics, data['image']

    def compute_G_loss(self, is_32, is_16, is_8, is_4, is_2, is_1, real_image, opt, trainable=True):
        G_losses = {}
        fake_image, KLD_loss = self.generate_fake(is_32, is_16, is_8, is_4, is_2, is_1, real_image, opt, compute_kld_loss=opt.use_vae, trainable=trainable)

        if opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(is_1, fake_image, real_image, trainable=False)

        G_losses['GAN'] = self.criterionGANLoss.loss(pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            # GAN_Feat_loss = np.empty(1).fill(0)
            GAN_Feat_loss = 0
            for i in range(num_D):
                num_intermediate_outputs = len(pred_fake[i]) -1
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeatureLoss(pred_fake[i][j], pred_real[i][j])
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feta / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVggLoss(fake_image, real_image) * self.opt.lambda_vgg


        return G_losses, fake_image

    def compute_D_loss(self, is_32, is_16, is_8, is_4, is_2, is_1, real_image):
        D_losses = {}
        # 不参加反向传播
        fake_image, _ = self.generate_fake(is_32, is_16, is_8, is_4, is_2, is_1, real_image, self.opt, trainable=False)

        pred_fake, pred_real = self.discriminate(is_1, fake_image, real_image, trainable=True)

        D_losses['D_fake'] = self.criterionGANLoss.loss(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGANLoss.loss(pred_real, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.E(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, is_32, is_16, is_8, is_4, is_2, is_1, real_image, opt, compute_kld_loss=False, trainable=True):
        z = None
        KLD_loss = None
        if opt.use_vae:
            z, mu, logvar = self.encode_z(real_image, trainable=trainable)
            if compute_kld_loss:
                KLD_loss = self.KLDloss(mu, logvar) * opt.lambda_kld
        fake_image = self.G(is_32, is_16, is_8, is_4, is_2, is_1, opt, z, trainable=trainable)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    def discriminate(self, input_semantics, fake_image, real_image, trainable=True):
        fake_concat = flow.concat([input_semantics, fake_image], 1, name='flow_cat_1')
        real_concat = flow.concat([input_semantics, real_image], 1, name='flow_cat_2')

        fake_and_real = flow.concat([fake_concat, real_concat], 0)
        discriminator_out = self.D(fake_and_real, trainable=trainable)

        pred_fake, pred_real = self.devide_pred(discriminator_out)

        return pred_fake, pred_real

    def devide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake_part = []
                real_part = []
                for tensor in p:
                    mid_index = tensor.shape[0]//2
                    fake_part.append(flow.slice(tensor, begin=[0, None, None, None], size=[mid_index, None, None, None]))
                    real_part.append(flow.slice(tensor, begin=[mid_index, None, None, None], size=[tensor.shape[0], None, None, None]))
                    # fake.append([tensor[:tensor.shape[0]//2]] for tensor in p)
                    # real.append([tensor[tensor.shape[0]//2:]] for tensor in p)
                fake.append(fake_part)
                real.append(real_part)
        else:
            fake = pred[:pred.shape[0]//2]
            real = pred[pred.shape[0]//2:]
        return fake, real


    def reparameterize(self, mu, logvar):
        std = flow.math.exp(0.5*logvar)
        ini = flow.random_normal_initializer()
        eps = flow.get_variable(
            shape=std.shape,
            initializer=ini,
        )
        return flow.math.multiply(eps, std) + mu

    def get_edge(self, t):
        edge = flow.zeros(t.size(), dtype=bytes)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def use_gpu(self):
        return self.opt.gpu_nums > 0