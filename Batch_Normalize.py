import torch
import torch.nn as nn


def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    #  测试模式下
    if not is_training:
        x_hat = (X - moving_mean) / torch.sqrt(moving_var+eps)
    #  训练模式下
    else:
        assert len(X.shape) in (2, 4)
        #  输入层为全连接的时候
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - moving_mean)**2).mean(dim=0)
        #  输入为二维卷积层的时候, N * C * H * W, 对C以外的维度求均值
        else:
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - moving_mean)**2).mean(dim=0, keepdim=True)\
                .mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        x_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var
        Y = gamma * x_hat + beta
        return Y, moving_mean, moving_var


class BatchNorm(nn.modules):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.ones(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
        self.training = True

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
        if self.moving_var.device != X.device:
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_mean, self.moving_var = batch_norm(self.training, X, self.gamma,
                                                          self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y
