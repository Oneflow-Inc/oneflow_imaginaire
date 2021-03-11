import numpy as np

import oneflow as flow
import oneflow.typing as tp

import generator as G
import discriminator as D
import loss as L

flow.env.init()
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_placement_scope(flow.scope.placement("gpu", "0:0"))

N, C, H, W = 2, 3, 256, 256

gen_config = dict(
    num_filters=64, 
    num_filters_mlp=256, 
    style_dims=64, 
    num_res_blocks=2,
    num_mlp_blocks=3, 
    num_downsamples_content=4, 
    num_downsamples_style=5, 
    num_image_channels=3, 
    weight_norm_type=""
)

dis_config = dict(
    num_filters=64, 
    max_num_filters=1024, 
    num_layers=6, 
    num_classes=119
)

loss_weights = dict(gan=1, image_recon=0.1, feature_matching=1)
learning_rate = 0.0001

@flow.global_function("train", func_config)
def TrainGenerator(
    images_content: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    images_style: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    labels_content: tp.Numpy.Placeholder((N,), dtype=flow.int32), 
    labels_style: tp.Numpy.Placeholder((N,), dtype=flow.int32)
):

    # generator
    content_a = G.ContentEncoder(
        images_content, 
        num_downsamples=gen_config["num_downsamples_content"], 
        num_res_blocks=gen_config["num_res_blocks"], 
        image_channels=gen_config["num_image_channels"], 
        num_filters=gen_config["num_filters"], 
        padding_mode="reflect", 
        activation_norm_type="instance", 
        weight_norm_type=gen_config["weight_norm_type"], 
        nonlinearity="relu"
    )

    style_a = G.StyleEncoder(
        images_content, 
        num_downsamples=gen_config["num_downsamples_style"], 
        num_filters=gen_config["num_filters"], 
        style_channels=gen_config["style_dims"], 
        padding_mode="reflect", 
        activation_norm_type="none", 
        weight_norm_type=gen_config["weight_norm_type"], 
        nonlinearity="relu"
    )

    style_b = G.StyleEncoder(
        images_style, 
        num_downsamples=gen_config["num_downsamples_style"], 
        num_filters=gen_config["num_filters"], 
        style_channels=gen_config["style_dims"], 
        padding_mode="reflect", 
        activation_norm_type="none", 
        weight_norm_type=gen_config["weight_norm_type"], 
        nonlinearity="relu"
    )

    style_b = flow.squeeze(style_b, axis=[2, 3])
    style_b = G.MLP(
        style_b, 
        output_dim=gen_config["num_filters_mlp"], 
        latent_dim=gen_config["num_filters_mlp"], 
        num_layers=gen_config["num_mlp_blocks"], 
        activation_norm_type="none", 
        nonlinearity="relu"
    )

    images_trans = G.Decoder(
        content_a, style_b, 
        num_enc_output_channels=content_a.shape[1], 
        num_image_channels=gen_config["num_image_channels"],
        num_upsamples=gen_config["num_downsamples_content"], 
        padding_type="reflect", 
        weight_norm_type=gen_config["weight_norm_type"], 
        nonlinearity="relu"
    )

    style_a = flow.squeeze(style_a, axis=[2, 3])
    style_a = G.MLP(
        style_a, 
        output_dim=gen_config["num_filters_mlp"], 
        latent_dim=gen_config["num_filters_mlp"], 
        num_layers=gen_config["num_mlp_blocks"], 
        activation_norm_type="none", 
        nonlinearity="relu"
    )

    images_recon = G.Decoder(
        content_a, style_a, 
        num_enc_output_channels=content_a.shape[1], 
        num_image_channels=gen_config["num_image_channels"],
        num_upsamples=gen_config["num_downsamples_content"], 
        padding_type="reflect", 
        weight_norm_type=gen_config["weight_norm_type"], 
        nonlinearity="relu"
    )

    # discriminator
    fake_out_trans, fake_features_trans = D.ResDiscriminator(
        images_trans, labels_style, 
        num_classes=dis_config["num_classes"], 
        num_filters=dis_config["num_filters"], 
        max_num_filters=dis_config["max_num_filters"], 
        num_layers=dis_config["num_layers"]
    )

    real_out_style, real_features_style = D.ResDiscriminator(
        images_style, labels_style
    )

    fake_out_recon, fake_features_recon = D.ResDiscriminator(
        images_recon, labels_content
    )

    # loss
    gan_loss = 0.5 * (
        L.GANloss(fake_out_trans, True, dis_update=False) + 
        L.GANloss(fake_out_recon, True, dis_update=False)
    )

    image_recon_loss = L.image_recon_loss(
        images_recon, images_content
    )
    
    feature_matching_loss = L.feature_matching_loss(
        fake_features_trans, real_features_style
    )

    loss = (
        gan_loss * loss_weights["gan"] + 
        image_recon_loss * loss_weights["image_recon"] + 
        feature_matching_loss * loss_weights["feature_matching"]
    )

    scheduler = flow.optimizer.PiecewiseConstantScheduler([], [learning_rate])
    flow.optimizer.Adam(scheduler, beta1=0.).minimize(loss)

    return images_trans, loss

@flow.global_function("train", func_config)
def TrainDiscriminator(
    images_trans: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    images_style: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    labels_style: tp.Numpy.Placeholder((N,), dtype=flow.int32)):

    # discriminator
    fake_out_trans, _ = D.ResDiscriminator(
        images_trans, labels_style
    )

    real_out_style, _ = D.ResDiscriminator(
        images_style, labels_style
    )

    loss = (
        L.GANloss(real_out_style, True, dis_update=True) + 
        L.GANloss(fake_out_trans, False, dis_update=True)
    )

    scheduler = flow.optimizer.PiecewiseConstantScheduler([], [learning_rate])
    flow.optimizer.Adam(scheduler, beta1=0.).minimize(loss)

    return loss

images_content = np.random.uniform(-10, 10, (N, C, H, W)).astype(np.float32)
images_style = np.random.uniform(-10, 10, (N, C, H, W)).astype(np.float32)
labels_content = np.random.uniform(0, 119, (N,)).astype(np.int32)
labels_style = np.random.uniform(0, 119, (N,)).astype(np.int32)

for epoch in range(10000):
    images_trans, loss_G = TrainGenerator(
        images_content, images_style, 
        labels_content, labels_style
    ).get()
    
    loss_D = TrainDiscriminator(
        images_trans.numpy(), images_style, labels_style
    ).get()
    
    print(
        f"[Epoch {epoch:4}] loss_G: {loss_G.numpy()}, loss_D: {loss_D.numpy()}"
    )
