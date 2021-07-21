import oneflow as flow
import numpy as np

def spectral_norm(w, iteration=1):
    w_shape = w.shape
    w = np.reshape(w, [-1, w_shape[0]])

    # u = flow.get_variable('u', shape=[1, w_shape[0]], initializer=flow.random_normal_initializer(), trainable=False)
    u = np.random.random([1, w_shape[0]])
    u_hat = u
    v_hat = None

    for i in range(iteration):
        v_ = np.matmul(u_hat, np.transpose(w))
        v_hat = v_/np.linalg.norm(v_)

        u_ = np.matmul(v_hat, w)
        u_hat = u_/np.linalg.norm(u_)

    sigma = np.matmul(np.matmul(v_hat, w), np.transpose(u_hat))

    w_norm = w/sigma
    w_norm = np.reshape(w_norm, w_shape)
    return w_norm