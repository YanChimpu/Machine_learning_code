#!/usr/bin/env python3
import numpy as np


class Conv_sliding_window(object):
    def __init__(self, input_data, weight_data, stride, padding="SAME"):
        self.input = np.asarray(input_data, np.float32)
        self.weight = np.asarray(weight_data, np.float32)
        self.stride = stride
        self.padding = padding

    def conv2d(self, c_out):
        """
        self.input: c * h * w
        self.weights: c * h * w
        """
        [c, w, h] = self.input.shape
        [kc, k, _] = self.weight.shape
        assert c == kc, "The count of channel in input is not equal to which in weight"
        output = np.zeros([c_out, int(h / self.stride), int(w / self.stride)], np.float32)
        # 分通道卷积
        for channel_out in range(c_out):
            for i in range(c):
                f_map = self.input[i, :, :]
                kernel = self.weight[i, :, :]
                output[channel_out, :, :] = self.compute_conv(f_map, kernel)
        return output

    def compute_conv(self, fm, kernel):
        [h, w] = fm.shape
        [k, _] = kernel.shape
        assert self.padding in ["SAME", "VALID", "FULL"]
        if self.padding == "SAME":
            pad_h = (self.stride * (h - 1) + k - h) // 2
            pad_w = (self.stride * (w - 1) + k - w) // 2
            rs_h, rs_w = h, w
        elif self.padding == "VALID":
            pad_h = 0
            pad_w = 0
            rs_h = (h - k) // self.stride + 1
            rs_w = (w - k) // self.stride + 1
        else:
            #  self.padding = "FULL"
            pad_w = k - 1
            pad_h = k - 1
            rs_h = (h + 2 * pad_h - k) // self.stride + 1
            rs_w = (w + 2 * pad_w - k) // self.stride + 1
        padding_fm = np.zeros([h + 2 * pad_h, w + 2 * pad_w], np.float32)
        padding_fm[pad_h:pad_h + h, pad_w:pad_w + w] = fm
        rs = np.zeros([rs_h, rs_w], np.float32)

        for i in range(rs_h):
            for j in range(rs_w):
                roi = padding_fm[i * self.stride: i * self.stride + k,
                                 j * self.stride: j * self.stride + k]
                rs[i, j] = np.sum(roi * kernel)
        return rs


if __name__=='__main__':
    input_data = [
        [
            [1, 0, 1, 2, 1],
            [0, 2, 1, 0, 1],
            [1, 1, 0, 2, 0],
            [2, 2, 1, 1, 0],
            [2, 0, 1, 2, 0],
        ],
        [
            [2, 0, 2, 1, 1],
            [0, 1, 0, 0, 2],
            [1, 0, 0, 2, 1],
            [1, 1, 2, 1, 0],
            [1, 0, 1, 1, 1],

        ],
    ]
    weight_data = [
        [
            [1, 0, 1],
            [-1, 1, 0],
            [0, -1, 0],
        ],
        [
            [-1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    ]
    conv = Conv_sliding_window(input_data, weight_data, 1, 'SAME')
    print(conv.conv2d(3))
