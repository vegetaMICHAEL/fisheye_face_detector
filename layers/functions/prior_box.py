import torch
from itertools import product as product
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']  # [16, 32], [64, 128], [256, 512]
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]  #'steps': [8, 16, 32],
        self.name = "s"

    def forward(self):
        anchors = []
        # 总结：每一个feature➡️每一个点➡️anchor的两种尺度（min_size）➡️所有anchor的中心点️
        for k, f in enumerate(self.feature_maps):  # [80, 80],[40, 40],[20, 20]
            min_sizes = self.min_sizes[k]
            # product(A,B)函数,返回A和B中的元素组成的笛卡尔积的元组,即A和B中的每一个元素两两配对
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    # 中心点的坐标：中心点x的位置加0.5然后乘上比例（不同尺度下，anchor步幅不同）
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]  # step = [8, 16, 32],
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
