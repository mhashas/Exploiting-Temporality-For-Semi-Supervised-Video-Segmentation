import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch.utils.model_zoo as model_zoo

from core.base_temporal_model import BaseTemporalModel

RESNET_101 = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = torch.cat(out, 1)
        return out


class PSPNet(BaseTemporalModel):
    def __init__(self, num_classes, args=None):
        super(PSPNet, self).__init__(args)
        self.pretrained = args.pretrained_resnet

        resnet = models.resnet101()

        if self.pretrained:
            resnet.load_state_dict(model_zoo.load_url(RESNET_101))

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        self.encoder_sequence_models = self.get_skip_sequence_models(args) if '+temporal' in self.sequence_model_type else None
        self.initialize_weights(self.ppm, self.final)


    def initialize_weights(self, *models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def get_skip_sequence_models(self, args):
        """
            Returns an array of inplace_abn f to model the temporal dimension of the skip connections
        """
        channels = [64,256,512,1024,2048]
        skip_connection_models = []
        for i in range(5):
            args.channels = channels[i] # a bit nasty @TODO is there a cleaner way?
            skip_connection_models.append(self.build_sequence_model(args))

        return nn.Sequential(*skip_connection_models)

    def get_number_channels(self, args):
        return args.channels

    def forward(self, x):
        x = self.remove_time_reshape(x)
        x_size = x.size()
        x = self.layer0(x)

        if self.encoder_sequence_models:
            x = self.temporal_forward(x, self.encoder_sequence_models[0])

        x = self.layer1(x)

        if self.encoder_sequence_models:
            x = self.temporal_forward(x, self.encoder_sequence_models[1])

        x = self.layer2(x)

        if self.encoder_sequence_models:
            x = self.temporal_forward(x, self.encoder_sequence_models[2])

        x = self.layer3(x)

        if self.encoder_sequence_models:
            x = self.temporal_forward(x, self.encoder_sequence_models[3])

        x = self.layer4(x)

        if self.encoder_sequence_models:
            x = self.temporal_forward(x, self.encoder_sequence_models[4])

        x = self.ppm(x)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        x = self.add_time_reshape(x)

        return x, None