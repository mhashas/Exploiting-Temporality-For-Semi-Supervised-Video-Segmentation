'''
Adapted work from https://github.com/jfzhang95/pytorch-deeplab-xception
'''

import torch
import torch.nn.functional as F
import torch.nn as nn

from core.resnet import ResNet50, ResNet101
from core.base_temporal_model import BaseTemporalModel

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride, norm_layer=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             norm_layer(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()

        low_level_inplanes = 256

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = norm_layer(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus(BaseTemporalModel):
    def __init__(self, args, num_classes=21, norm_layer=nn.BatchNorm2d):
        super(DeepLabv3_plus, self).__init__(args)
        self.pretrained = args.pretrained_resnet

        if args.model == 'deeplab':
            self.backbone = ResNet101(args.output_stride, norm_layer=norm_layer, pretrained=self.pretrained, args=args)
        elif args.model == 'deeplab-50':
            self.backbone = ResNet50(args.output_stride, norm_layer=norm_layer, pretrained=self.pretrained, args=args)
        else:
            raise NotImplementedError

        self.aspp = ASPP(args.output_stride, norm_layer=norm_layer)
        self.sequence_model_low, self.sequence_model_high = None, None
        #self.sequence_model_low = self.build_sequence_model(args) if 'sequence' in self.mode and not '+' in self.sequence_model_type else None
        #self.sequence_model_high = self.build_sequence_model(args) if 'sequence' in self.mode else None
        self.decoder = Decoder(num_classes, norm_layer=norm_layer)

    def forward(self, input):
        x = self.remove_time_reshape(input)
        x, low_level_feat = self.backbone(x)
        x = self.aspp(x)

        if self.sequence_model_high:
            x = self.temporal_forward(x, self.sequence_model_high)

        if self.sequence_model_low:
            low_level_feat = self.temporal_forward(low_level_feat, self.sequence_model_low)

        x = self.decoder(x, low_level_feat)

        interpolate_size = input.size()[3:] if 'sequence' in self.mode else input.size()[2:]
        x = F.interpolate(x, size=interpolate_size, mode='bilinear', align_corners=True)
        x = self.add_time_reshape(x)

        return x, None


    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]

        if self.sequence_model_low:
            modules.append(self.sequence_model_low)
        if self.sequence_model_high:
            modules.append(self.sequence_model_high)

        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p

    def get_train_parameters(self, lr):
        if self.pretrained:
            train_params = [{'params': self.get_1x_lr_params(), 'lr': lr / 10},
                            {'params': self.get_10x_lr_params(), 'lr': lr}]
        else:
            train_params = [{'params': self.parameters(), 'lr': lr}]

        return train_params