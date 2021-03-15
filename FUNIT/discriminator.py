import numpy as np
import oneflow as flow
import utils


def ResDiscriminator(
    input, labels=None, num_classes=119, num_filters=64, max_num_filters=1024, 
    num_layers=6, padding_mode="reflect", weight_norm_type="", 
    name="ResDiscriminator", trainable=None
):

    assert padding_mode == "reflect"
    assert weight_norm_type == ""

    namer = utils.namer_factory()

    def Res2dBlock(input, num_filters):
        out = input

        for _ in range(2):
            out = flow.nn.leaky_relu(out, alpha=0.2)
            out = flow.reflection_pad2d(out, padding=1)
            out = utils.conv2d_layer(
                out, num_filters, 3, 1, "VALID", 
                name=namer("conv"), trainable=trainable
            )

        if input.shape == out.shape:
            return input + out
        else:
            shortcut_out = utils.conv2d_layer(
                input, num_filters, 1, 1, "VALID", 
                name=namer("conv"), trainable=trainable
            )
            return shortcut_out + out

    out = input
    with flow.scope.namespace(name):
        out = flow.reflection_pad2d(out, padding=3)
        out = utils.conv2d_layer(
            out, num_filters, 7, 1, "VALID", 
            name=namer("conv"), trainable=trainable
        )
        
        for i in range(num_layers):
            out = Res2dBlock(out, num_filters)
            num_filters = min(num_filters * 2, max_num_filters)
            out = Res2dBlock(out, num_filters)

            if i != num_layers - 1:
                out = flow.reflection_pad2d(out, padding=1)
                out = flow.nn.avg_pool2d(
                    out, ksize=3, strides=2, padding="VALID"
                )

        features = out

        features_1x1 = flow.reshape(features, [*features.shape[:2], -1])
        features_1x1 = flow.math.reduce_mean(features_1x1, 2)

        if labels is None:
            return features_1x1

        embeddings = utils.embedding_layer(
            labels, num_classes, num_filters, 
            name=namer("embedding"), trainable=trainable
        )

        outputs = flow.nn.leaky_relu(features, alpha=0.2)
        outputs = utils.conv2d_layer(
            outputs, 1, 1, 1, "VALID", 
            name=namer("conv"), trainable=trainable
        )
        feat = flow.math.reduce_sum(
            embeddings * features_1x1, axis=1, keepdims=True
        )
        outputs += flow.reshape(feat, shape=(input.shape[0], 1, 1, 1))

        return outputs, features_1x1


if __name__ == "__main__":
    from typing import Tuple
    import oneflow.typing as tp

    batch = 1
    width = 128

    @flow.global_function(type="predict")
    def test_job1(
        images: tp.Numpy.Placeholder((batch, 3, width, width), dtype=flow.float32),
    ) -> tp.Numpy:

        with flow.scope.placement("gpu", "0:0"):
            return ResDiscriminator(images)

    @flow.global_function(type="predict")
    def test_job2(
        images: tp.Numpy.Placeholder((batch, 3, width, width), dtype=flow.float32),
        labels: tp.Numpy.Placeholder((batch,), dtype=flow.int32)
    ) -> Tuple[tp.Numpy, tp.Numpy]:

        with flow.scope.placement("gpu", "0:0"):
            return ResDiscriminator(images, labels)

    images = np.random.uniform(-10, 10, (batch, 3, width, width)).astype(np.float32)
    out = test_job1(images)

    print(out.shape) # (1, 1024)

    images = np.random.uniform(-10, 10, (batch, 3, width, width)).astype(np.float32)
    labels = np.random.uniform(0, 119, (batch,)).astype(np.int32)
    out1, out2 = test_job2(images, labels)

    print(out1.shape, out2.shape)
