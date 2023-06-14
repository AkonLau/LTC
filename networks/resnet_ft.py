import torch
import torch.nn as nn
from torchvision import models
class ResNet(nn.Module):

    def __init__(self,conf):
        super(ResNet, self).__init__()
        basenet = eval('models.'+conf.netname)(pretrained=conf.pretrained)
        self.conv3 = nn.Sequential(*list(basenet.children())[:-4])
        self.conv4 = list(basenet.children())[-4]
        self.conv5 = list(basenet.children())[-3]

        self.midlevel = False
        self.isdetach = True
        if 'midlevel' in conf:
            self.midlevel = conf.midlevel
        if 'isdetach' in conf:
            self.isdetach = conf.isdetach

        mid_dim = 1024
        feadim = 2048
        if conf.netname in ['resnet18','resnet34']:
            mid_dim = 256
            feadim = 512
        elif conf.netname in ['resnet32']:
            mid_dim = 32
            feadim = 64
        if self.midlevel:
            self.mcls = nn.Linear(mid_dim, conf.num_class)
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.conv4_1 = nn.Sequential(nn.Conv2d(mid_dim, mid_dim, 1, 1), nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feadim, conf.num_class)

        # semantic layer for auxiliary target coding regularization
        self.target_coding = conf.HTC or conf.LTC
        self.code_length = conf.code_length
        if self.target_coding is True:
            self.hash_layer = nn.Sequential(
                nn.Linear(feadim, feadim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(feadim, feadim),
                nn.ReLU(),
                nn.Linear(feadim, self.code_length),
            )

    def forward(self, x):
        conv3 = self.conv3(x)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        fea_pool = self.avg_pool(conv5).view(conv5.size(0), -1)
        logits = self.classifier(fea_pool)

        if self.target_coding:
            # target coding
            xlocal_attr = self.hash_layer(fea_pool)
        else:
            xlocal_attr = None

        if self.midlevel:
            if self.isdetach:
                conv4_1 = conv4.detach()
            else:
                conv4_1 = conv4
            conv4_1 = self.conv4_1(conv4_1)
            pool4_1 = self.max_pool(conv4_1).view(conv4_1.size(0),-1)
            mlogits = self.mcls(pool4_1)
        else:
            mlogits = None
            pool4_1 = None
            conv4_1 = None

        return logits,[conv5,conv4_1],mlogits,[fea_pool, pool4_1], xlocal_attr


    def _init_weight(self, block):
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_params(self, param_name):
        ftlayer_params = list(self.conv3.parameters()) +\
                           list(self.conv4.parameters()) +\
                           list(self.conv5.parameters())
        ftlayer_params_ids = list(map(id, ftlayer_params))
        freshlayer_params = filter(lambda p: id(p) not in ftlayer_params_ids, self.parameters())

        return eval(param_name+'_params')

def get_net(conf):
    return ResNet(conf)

if __name__ == '__main__':
    import random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=200, help="class numbers")
    parser.add_argument('--netname', default='resnet50', type=str, help='config files')
    parser.add_argument('--pretrained', default=0, type=float, help='loss weights')

    conf = parser.parse_args()
    model = get_net(conf).cuda()
    xf = torch.rand((2, 3, 448, 448)).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print('Total parameters: %d' % (total_params))

    out = model(xf)
