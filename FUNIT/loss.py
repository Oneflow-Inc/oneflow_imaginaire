import oneflow as flow

def GANloss(dis_output, t_real, gan_mode="hinge", dis_update=True):
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

def image_recon_loss(gen_recon, src_content):
    return flow.nn.L1Loss(gen_recon, src_content)

def feature_matching_loss(fake_features_trans, real_features_style):
    return flow.nn.L1Loss(fake_features_trans, real_features_style)
