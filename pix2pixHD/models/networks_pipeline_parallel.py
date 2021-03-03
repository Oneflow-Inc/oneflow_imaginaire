import oneflow as flow
import numpy as np

def conv2d_layer(
    name,
    input,
    out_channel,
    kernel_size = 3,
    strides = 1,
    padding = "SAME", # or [[], [], [], []]
    data_format = "NCHW",
    dilation_rate = 1,
    use_bias = True,
    weight_initializer = flow.random_normal_initializer(mean = 0.0, stddev = 0.02),
    bias_initializer = flow.zeros_initializer(),
    trainable = True,
    reuse = True 
):   
    weight_shape = (out_channel, input.shape[1], kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "_weight",
        shape = weight_shape,
        dtype = input.dtype,
        initializer = weight_initializer,
        trainable = trainable,
        reuse = reuse
    )
    output = flow.nn.conv2d(input, weight, strides, padding, data_format, dilation_rate)
    if use_bias:
        bias = flow.get_variable(
            name + "_bias",
            shape = (out_channel,),
            dtype = input.dtype,
            initializer = bias_initializer,
            trainable = trainable
        )
        output = flow.nn.bias_add(output, bias, data_format)
    return output

def upsampleConvLayer(
    input,
    name_prefix,
    channel,
    kernel_size,
    hw_scale = (2, 2), 
    data_format = "NCHW", 
    interpolation = "bilinear",
    # interpolation = "nearest",
    trainable = True):
        upsample = flow.layers.upsample_2d(input, size = hw_scale, data_format = data_format, interpolation = interpolation, name = name_prefix + "_%s" % interpolation)
        return conv2d_layer(name_prefix + "_conv", upsample, channel, kernel_size = kernel_size, strides = 1, trainable = trainable)


def deconv(input, out_channel, name_prefix, kernel_size = 4, strides = [2, 2], trainable = True, reuse = True):
    weight = flow.get_variable(
        name_prefix + "_weight",
        shape = (input.shape[1], out_channel, kernel_size, kernel_size),
        dtype = flow.float,
        initializer = flow.random_normal_initializer(mean = 0.0, stddev = 0.02),
        trainable = trainable,
        reuse = reuse
    )
    return flow.nn.conv2d_transpose(
                input,
                weight,
                strides = strides,
                padding = "SAME",
                output_shape = (input.shape[0], out_channel, input.shape[2] * strides[0], input.shape[3] * strides[1]))

def norm_layer(input, name, norm_type="instance", trainable = True, reuse = True):
    return flow.nn.InstanceNorm2d(input, eps=1e-5, affine=False)

def ResnetBlock(input, name_prefix, dim, norm_type="instance", 
                use_dropout=False, trainable=True, reuse=True):
    out = flow.reflection_pad2d(input, padding=[1, 1, 1, 1])
    out = conv2d_layer(name_prefix + "_conv1", out, dim, kernel_size=3, padding="VALID", trainable=trainable, reuse=reuse)
    out = norm_layer(out, name_prefix + "_norm1", norm_type=norm_type, trainable=trainable, reuse=reuse)
    out = flow.nn.relu(out)
    if use_dropout:
        out = flow.nn.dropout(out, rate=0.5)

    out = flow.reflection_pad2d(out, padding=[1, 1, 1, 1])
    out = conv2d_layer(name_prefix + "_conv2", out, dim, kernel_size=3, padding="VALID", trainable=trainable, reuse=reuse)
    out = norm_layer(out, name_prefix + "_norm2", norm_type=norm_type, trainable=trainable, reuse=reuse)

    return input + out

def GlobalGenerator(input, var_name_prefix, output_nc, ngf=64, n_downsampling=3, 
                    n_blocks=9, norm_type="instance", trainable=True, reuse=True, return_before_final_conv=False):
    with flow.scope.namespace(var_name_prefix):
        out = flow.reflection_pad2d(input, padding=[3, 3, 3, 3])
        out = conv2d_layer("conv1", out, ngf, kernel_size=7, padding="VALID", trainable=trainable, reuse=reuse)
        out = norm_layer(out, "norm1", norm_type=norm_type, trainable=trainable, reuse=reuse)
        out = flow.nn.relu(out)
    
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            out = conv2d_layer("conv_downsample_%d" % i, out, ngf * mult * 2, 
                                kernel_size=3, strides=2, padding="SAME", trainable=trainable, reuse=reuse)
            out = norm_layer(out, "norm_downsample_%d" % i, norm_type=norm_type, trainable=trainable, reuse=reuse)
            out = flow.nn.relu(out)
        
        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            out = ResnetBlock(out, "resblock_%d" % i, ngf * mult, norm_type=norm_layer, trainable=trainable, reuse=reuse)

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            out = deconv(out, int(ngf * mult / 2), "deconv_%d" % i, kernel_size=3, strides=[2, 2], trainable=trainable, reuse=reuse)
            out = norm_layer(out, "norm_upsample_%d" % i, norm_type=norm_type, trainable=trainable, reuse=reuse)
            out = flow.nn.relu(out)
        
        if return_before_final_conv:
            return out

        out = flow.reflection_pad2d(out, padding=[3, 3, 3, 3])
        out = conv2d_layer("conv_last", out, output_nc, kernel_size=7, padding="VALID", trainable=trainable, reuse=reuse)
        out = flow.math.tanh(out)

        return out

def LocalEnhancer(input, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                  n_blocks_local=3, norm_type="instance",
                  trainable=True, reuse=True, train_global_generator=True):
        output_prev = 0
        if train_global_generator:
            with flow.scope.placement("gpu", "0:1"):
                input_downsampled = flow.nn.max_pool2d(input, 3, 2, "SAME")
                ### output at coarest level, get rid of final convolution layers
                output_prev = GlobalGenerator(input_downsampled, "G1", output_nc,
                                            ngf=ngf*2, n_downsampling=n_downsample_global, n_blocks=n_blocks_global, 
                                            norm_type=norm_type, trainable=trainable, reuse=reuse, return_before_final_conv=True)       

        with flow.scope.namespace("G2"):
            with flow.scope.placement("gpu", "0:1"):
                ### downsample            
                out = flow.reflection_pad2d(input, padding=[3, 3, 3, 3])
                out = conv2d_layer("conv1", out, ngf, kernel_size=7, padding="VALID", trainable=trainable, reuse=reuse)
                out = norm_layer(out, "norm1", norm_type=norm_type, trainable=trainable, reuse=reuse)
                out = flow.nn.relu(out)

                out = conv2d_layer("conv_downsample", out, ngf*2, 
                                kernel_size=3, strides=2, padding="SAME", trainable=trainable, reuse=reuse)
                out = norm_layer(out, "norm_downsample", norm_type=norm_type, trainable=trainable, reuse=reuse)
                out = flow.nn.relu(out)

                if train_global_generator:
                    out = out + output_prev

            with flow.scope.placement("gpu", "0:1"): # for CelebA-HQ
            # with flow.scope.placement("gpu", "0:2"): # for cityscapes
                ### residual blocks
                for i in range(n_blocks_local):
                    out = ResnetBlock(out, "resblock_%d" % i, ngf * 2, norm_type=norm_layer, trainable=trainable, reuse=reuse)

                ### upsample
                out = deconv(out, ngf, "deconv", kernel_size=3, strides=[2, 2], trainable=trainable, reuse=reuse)
                out = norm_layer(out, "norm_upsample", norm_type=norm_type, trainable=trainable, reuse=reuse)
                out = flow.nn.relu(out)

                ### final convolution
                out = flow.reflection_pad2d(out, padding=[3, 3, 3, 3])
                out = conv2d_layer("conv_last", out, output_nc, kernel_size=7, padding="VALID", trainable=trainable, reuse=reuse)
                out = flow.math.tanh(out)

                return out

def define_G(input, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9,
             n_blocks_local=3, norm_type='instance', trainable=True, reuse=True, train_global_generator=True):    
    if netG == 'global':    
        netG = GlobalGenerator(input, "G1", output_nc,
                              ngf=ngf, n_downsampling=n_downsample_global, n_blocks=n_blocks_global, 
                              norm_type=norm_type, trainable=trainable, reuse=reuse)  
    elif netG == 'local':
        netG = LocalEnhancer(input, output_nc, ngf=ngf, n_downsample_global=n_downsample_global,
                             n_blocks_global=n_blocks_global, n_blocks_local=n_blocks_local, norm_type=norm_type,
                             trainable=trainable, reuse=reuse, train_global_generator=train_global_generator)
    elif netG == 'encoder':
        raise('generator not implemented!')
    else:
        raise('generator not implemented!')
    return netG


def MultiscaleRecLoss(fake, real, num_D=3):
    real_downsampled = real
    fake_downsampled = fake
    loss = 0
    for i in range(num_D):
        loss = flow.nn.L1Loss(real_downsampled, fake_downsampled) + loss
        if i != (num_D-1):
            real_downsampled = flow.nn.avg_pool2d(real_downsampled, 3, 2, "SAME")
            fake_downsampled = flow.nn.avg_pool2d(fake_downsampled, 3, 2, "SAME")
    return loss

def MultiscaleDiscriminator(input, ndf=64, n_layers=3, norm_type="instance",
                            use_sigmoid=False, num_D=3, trainable=True, reuse=True):
    with flow.scope.namespace("Multiscale_"):
        input_downsampled = input
        result = []
        for i in range(num_D):
            out = NLayerDiscriminator(input_downsampled, "D_%d" % i, ndf=ndf, n_layers=n_layers,
                                    norm_type=norm_type, use_sigmoid=use_sigmoid,
                                    trainable=trainable, reuse=reuse)
            result.append(out)
            if i != (num_D-1):
                input_downsampled = flow.nn.avg_pool2d(input_downsampled, 3, 2, "SAME")
    return result
        
# Defines the PatchGAN discriminator with the specified arguments.
def NLayerDiscriminator(input, name_prefix, ndf=64, n_layers=3, norm_type="instance",
                        use_sigmoid=False, trainable=True, reuse=True):
    with flow.scope.namespace(name_prefix):
        res = []

        kernel_size = 4
        padw = int(np.ceil((kernel_size-1.0)/2))
        padding = [[0, 0], [0, 0], [padw, padw], [padw, padw]]
        out = conv2d_layer("conv_0", input, ndf, kernel_size=kernel_size, strides=2,
                            padding=padding, trainable=trainable, reuse=reuse)
        out = flow.nn.leaky_relu(out, 0.2)
        res.append(out)

        nf = ndf
        for i in range(1, n_layers):
            nf = min(nf * 2, 512)
            out = conv2d_layer("conv_downsample_%d" % i, out, nf, kernel_size=kernel_size,
                               strides=2, padding=padding, trainable=trainable, reuse=reuse)
            out = norm_layer(out, "norm_downsample_%d" % i, norm_type=norm_type, trainable=trainable, reuse=reuse)
            out = flow.nn.leaky_relu(out, 0.2)
            res.append(out)

        nf = min(nf * 2, 512)
        out = conv2d_layer("conv_1", out, nf, kernel_size=kernel_size, strides=1,
                           padding=padding, trainable=trainable, reuse=reuse)
        out = norm_layer(out, "norm_1", norm_type=norm_type, trainable=trainable, reuse=reuse)
        out = flow.nn.leaky_relu(out, 0.2)
        res.append(out)

        out = conv2d_layer("last_conv", out, 1, kernel_size=kernel_size, strides=1,
                           padding=padding, trainable=trainable, reuse=reuse)
        res.append(out)

        if use_sigmoid:
            out = flow.math.sigmoid(out)
            res.append(out)

    return res

def GANLoss(input, target_is_real, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
    assert isinstance(input[0], list)
    loss = 0
    for i in range(0, len(input)):
        if target_is_real:
            target = flow.constant_like(input[i][-1], target_real_label)
        else:
            target = flow.constant_like(input[i][-1], target_fake_label)
        if use_lsgan:
            loss = flow.nn.MSELoss(input[i][-1], target) + loss
        else:
            loss = flow.nn.BCELoss(input[i][-1], target) + loss
    return loss

