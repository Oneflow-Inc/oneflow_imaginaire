import oneflow as flow
import numpy as np
from models.vgg16 import vgg16bn_style_layer

class GAN_loss():
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0, opt=None):
        super(GAN_loss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.gan_mode = gan_mode
        self.opt = opt
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None

        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan mode'+str(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = np.empty(1).fill(self.real_label)
            return flow.expand(self.real_label_tensor, input.shape)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = np.empty(1).fill(self.fake_label)
            return flow.expand(self.fake_label_tensor, input.shape)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = flow.ones([1])
        return flow.expand(self.zero_tensor, input.shape)


    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return flow.experimental.nn.BCEWithLogitsLoss(input, target_tensor)
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return flow.nn.MSELoss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    loss = 0
                    femmu = 0  # 用于求平均
                    if type(input) is list:
                        for input_ in input:
                            if type(input_) is list:
                                for input__ in input_:
                                    minval = flow.math.minimum(input__ - flow.ones_like(input__),
                                                               self.get_zero_tensor(input__))
                                    loss += flow.math.negative(flow.math.reduce_mean(minval))
                                    femmu +=1
                            else:
                                minval = flow.math.minimum(input_ - flow.ones_like(input_), self.get_zero_tensor(input_))
                                loss += flow.math.negative(flow.math.reduce_mean(minval))
                    else:
                        loss += flow.math.negative(flow.math.reduce_mean(input))
                else:
                    loss = 0
                    femmu = 0  # 用于求平均
                    if type(input) is list:
                        for input_ in input:
                            if type(input_) is list:
                                for input__ in input_:
                                    minval = flow.math.minimum(flow.math.negative(input__) - flow.ones_like(input__), self.get_zero_tensor(input__))
                                    loss += flow.math.negative(flow.math.reduce_mean(minval))
                                    femmu+=1
                            else:
                                minval = flow.math.minimum(flow.math.negative(input_) - flow.ones_like(input_), self.get_zero_tensor(input_))
                                loss += flow.math.negative(flow.math.reduce_mean(minval))
                    else:
                        loss += flow.math.negative(flow.math.reduce_mean(input))
            else:
                loss = 0
                femmu = 0 #用于求平均
                if type(input) is list:
                    for input_ in input:
                        if type(input_) is list:
                            for input__ in input_:
                                loss += flow.math.negative(flow.math.reduce_mean(input__))
                                femmu+=1
                        else:
                            loss += flow.math.negative(flow.math.reduce_mean(input_))
                else:
                    loss += flow.math.negative(flow.math.reduce_mean(input))
            return loss/femmu
        else:
            raise ('Have not implemet!')


def gan_loss(dis_output, t_real, gan_mode="hinge", dis_update=True):
    assert gan_mode == "hinge"

    if dis_update:
        if t_real:
            loss = flow.math.minimum(
                dis_output - flow.ones_like(dis_output),
                flow.zeros_like(dis_output)
            )
            loss = flow.math.negative(flow.math.reduce_mean(loss))
        else:
            loss = flow.math.minimum(
                flow.math.negative(dis_output) - flow.ones_like(dis_output),
                flow.zeros_like(dis_output)
            )
            loss = flow.math.negative(flow.math.reduce_mean(loss))
    else:
        loss = flow.math.negative(flow.math.reduce_mean(dis_output))

    return loss


class VGG_loss():
    def __init__(self):
        super(VGG_loss, self).__init__()
        self.vgg = vgg16bn_style_layer
        self.criterion = flow.nn.L1Loss
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class KLD_loss():
    def __init__(self, ):
        super(KLD_loss, self).__init__()

    def kldloss(self, mu, logvar):
        return -0.5*flow.summary(1+logvar-flow.math.pow(mu, 2)-flow.math.exp(logvar))

    def __call__(self, *args, **kwargs):
        return self.kldloss(*args, **kwargs)