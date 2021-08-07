import oneflow as flow
from .network.normalization import spadeRes, conv2d_layer, deconv


def generator(is_32, is_16, is_8, is_4, is_2, is_1, opt, z=None, trainable=True):
    spectral = False
    init = flow.xavier_uniform_initializer()

    def up(d, name, trainable):
        if opt.up == 'nearest':
            return flow.layers.upsample_2d(d, name=name)
        elif opt.up == 'bilinear':
            return flow.layers.upsample_2d(d, name=name, interpolation='bilinear')
        elif opt.up == 'deconv':
            return deconv(d, d.shape[1], name_prefix=name, trainable=trainable)
        elif opt.up == 'subpixel':
            raise NotImplementedError

    if opt.use_vae:
        init_z_method = flow.random_normal_initializer()
        if z is None:
            random_z = flow.get_variable(
                name='a+',
                # shape=(seg.shape[0], 256),
                # dtype=seg.dtype,
                initializer=init_z_method,
                trainable=trainable
            )
        d = flow.layers.dense(random_z, 16384, trainable=trainable)
        d = flow.reshape(d, shape=[-1, 1024, 4, 4])
    else:
        seg_ = is_32
        d = flow.layers.conv2d(seg_, 16*64, kernel_size=3, padding='SAME', name='g_conv2d_2', trainable=trainable, kernel_initializer=init)


    # head_0
    d = spadeRes(d, is_32, 1024, spectral=spectral, trainable=trainable, name_prefix='g_spadeRes_1')
    d = up(d, name='g_u1', trainable=trainable)
    # head_1
    d = spadeRes(d, is_16, 1024, spectral=spectral, trainable=trainable, name_prefix='g_spadeRes_2')
    # d = up(d)
    # head_2
    d = spadeRes(d, is_16, 1024, spectral=spectral, trainable=trainable, name_prefix='g_spadeRes_3')
    d = up(d, name='g_u2', trainable=trainable)
    # head_3
    d = spadeRes(d, is_8, 512, spectral=spectral, trainable=trainable, name_prefix='g_spadeRes_4')
    d = up(d, name='g_u3', trainable=trainable)
    # head_4
    d = spadeRes(d, is_4, 256, spectral=spectral, trainable=trainable, name_prefix='g_spadeRes_5')
    d = up(d, name='g_u4', trainable=trainable)
    # head_5
    d = spadeRes(d, is_2, 128, spectral=spectral, trainable=trainable, name_prefix='g_spadeRes_6')
    d = up(d, name='g_u5', trainable=trainable)
    # head_6
    d = spadeRes(d, is_1, 64, spectral=spectral, trainable=trainable, name_prefix='g_spadeRes_7')
    # d = up(d)
    d = flow.nn.leaky_relu(d)
    d = conv2d_layer(d, 3, kernel_size=3, trainable=trainable, name='g_conv2d_3', padding='SAME')
    # d = flow.experimental.tanh(d)
    d = flow.math.tanh(d)
    return d
