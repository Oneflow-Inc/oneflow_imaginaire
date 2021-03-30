from functools import partial
import os
import numpy as np

import oneflow as flow
import oneflow.typing as tp

import generator as G
import discriminator as D
import loss as L

import dataset
import options

if __name__ == "__main__":
    flow.config.enable_legacy_model_io(False)

    flow.env.init()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())
    # func_config.default_placement_scope(flow.scope.placement("gpu", "0:0-1"))

    opt = options.BaseOptions().parse()

    num_gpus = opt.num_gpus
    # assert num_gpus in [1, 2]
    flow.config.gpu_device_num(num_gpus)

    N = opt.batch_size
    C = opt.num_image_channels
    H = opt.image_size
    W = opt.image_size

    gen_config = dict(
        num_filters=opt.gen_num_filters, 
        num_filters_mlp=opt.gen_num_filters_mlp, 
        style_dims=opt.gen_style_dims, 
        num_res_blocks=opt.gen_num_res_blocks,
        num_mlp_blocks=opt.gen_num_mlp_blocks, 
        num_downsamples_content=opt.gen_num_downsamples_content, 
        num_downsamples_style=opt.gen_num_downsamples_style, 
        num_image_channels=opt.num_image_channels, 
        weight_norm_type=""
    )

    dis_config = dict(
        num_filters=opt.dis_num_filters, 
        max_num_filters=opt.dis_max_num_filters, 
        num_layers=opt.dis_num_layers, 
        num_classes=opt.label_nc
    )

    loss_weights = dict(
        gan=opt.gan_weight, 
        image_recon=opt.image_recon_weight, 
        feature_matching=opt.feature_matching_weight
    )

    learning_rate = opt.lr

    @flow.global_function("train", func_config)
    def TrainGenerator(
        images_content: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
        images_style: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
        labels_content: tp.Numpy.Placeholder((N,), dtype=flow.int32), 
        labels_style: tp.Numpy.Placeholder((N,), dtype=flow.int32)
    ):

        # with flow.scope.placement("gpu", "0:0"):
        with flow.scope.consistent_view():
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

        # with flow.scope.placement("gpu", f"0:{num_gpus-1}"):
        with flow.scope.consistent_view():
            # discriminator
            fake_out_trans, fake_features_trans = D.ResDiscriminator(
                images_trans, labels_style, 
                num_classes=dis_config["num_classes"], 
                num_filters=dis_config["num_filters"], 
                max_num_filters=dis_config["max_num_filters"], 
                num_layers=dis_config["num_layers"], 
                trainable=False
            )

            real_out_style, real_features_style = D.ResDiscriminator(
                images_style, labels_style, 
                num_classes=dis_config["num_classes"], 
                num_filters=dis_config["num_filters"], 
                max_num_filters=dis_config["max_num_filters"], 
                num_layers=dis_config["num_layers"], 
                trainable=False
            )

            fake_out_recon, fake_features_recon = D.ResDiscriminator(
                images_recon, labels_content, 
                num_classes=dis_config["num_classes"], 
                num_filters=dis_config["num_filters"], 
                max_num_filters=dis_config["max_num_filters"], 
                num_layers=dis_config["num_layers"], 
                trainable=False
            )

            # loss
            gan_loss = 0.5 * (
                L.GANloss(fake_out_trans, t_real=True, dis_update=False) + 
                L.GANloss(fake_out_recon, t_real=True, dis_update=False)
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
        labels_style: tp.Numpy.Placeholder((N,), dtype=flow.int32)
    ):

        # with flow.scope.placement("gpu", f"0:{num_gpus-1}"):
        with flow.scope.consistent_view():
            # discriminator
            fake_out_trans, _ = D.ResDiscriminator(
                images_trans, labels_style, 
                num_classes=dis_config["num_classes"], 
                num_filters=dis_config["num_filters"], 
                max_num_filters=dis_config["max_num_filters"], 
                num_layers=dis_config["num_layers"], 
                trainable=True
            )

            real_out_style, _ = D.ResDiscriminator(
                images_style, labels_style, 
                num_classes=dis_config["num_classes"], 
                num_filters=dis_config["num_filters"], 
                max_num_filters=dis_config["max_num_filters"], 
                num_layers=dis_config["num_layers"], 
                trainable=True
            )

            loss = (
                L.GANloss(real_out_style, t_real=True, dis_update=True) + 
                L.GANloss(fake_out_trans, t_real=False, dis_update=True)
            )

            scheduler = flow.optimizer.PiecewiseConstantScheduler([], [learning_rate])
            flow.optimizer.Adam(scheduler, beta1=0.).minimize(loss)

            return loss

    print("Job function configured")

    augment = partial(
        dataset.augment, 
        random_scale_limit=opt.random_scale_limit, 
        resize_smallest_side=opt.resize_smallest_side, 
        random_crop_h_w=opt.image_size
    )
    train_dataset = dataset.Dataset(os.path.join(opt.dataset_dir, "train"), augment=augment)
    print("Dataset loaded")

    for epoch in range(opt.epoch):
        data_iter = train_dataset.data_iterator(N)
        iteration = 0

        while True:
            try:
                images_content, labels_content = next(data_iter)
                images_style, labels_style = next(data_iter)
            except StopIteration:
                break

            images_trans, loss_G = TrainGenerator(
                images_content, images_style, 
                labels_content, labels_style
            ).get()

            loss_D = TrainDiscriminator(
                images_trans.numpy(), images_style, labels_style
            ).get()

            if iteration % 5 == 0:
                loss_G_data = f"{loss_G.numpy()[0]}"
                loss_D_data = f"{loss_D.numpy()[0]}"

                print(
                    f"[Epoch {epoch:4}] loss_G: {loss_G_data}, loss_D: {loss_D_data}"
                )

                if iteration % 300 == 0:
                    filename = (
                        f"{opt.checkpoints_dir}/"
                        f"epoch_{epoch}_iter_{iteration}_"
                        f"Gloss_{loss_G_data}_Dloss_{loss_D_data}"
                    )

                    flow.checkpoint.save(filename)
                    print(f"checkpoint saved to {filename}")

            iteration += 1

        train_dataset.shuffle()
