import torchvision
import torch.nn as nn
import torch
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import model_urls, conv1x1


class BasicBlock(torchvision.models.resnet.BasicBlock):
    """
    reimplementation of basic block with dropout layer after each conv layer
    """
    def __init__(self,*args, dropout_p=0.2,**kwargs):
        super(BasicBlock, self).__init__(*args, **kwargs)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(torchvision.models.resnet.Bottleneck):
    """
    reimplementation of bottleneck block with dropout layer after each conv layer
    """
    def __init__(self, *args, dropout_p=0.2, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.dropout3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, *args, dropout_p=0.2, **kwargs):
        self.dropout_p = dropout_p
        super(ResNet, self).__init__(*args, **kwargs)
        self.dropout1 = nn.Dropout(p=dropout_p)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.Dropout(p=self.dropout_p),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dropout_p=self.dropout_p))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, dropout_p=self.dropout_p))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _resnet(arch, block, layers, pretrained, progress, dropout_p=0.2, **kwargs):
    model = ResNet(block, layers, dropout_p=dropout_p, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        #modify the pretrained weights as the structure of downsample module (nn.sequential) is modified
        state_dict_new = dict(state_dict)                  
        for key in list(state_dict_new):
            names = key.split(".")
            if len(names) >=4 and (names[2] == "downsample") and (names[3] == "1"):
                names[3] = "2"
                new_key = ".".join(names)
                state_dict_new[new_key] = state_dict_new[key]
                del state_dict_new[key]
        model.load_state_dict(state_dict_new)
    return model

def resnet18(pretrained=False, progress=True, dropout_p=0.2, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,dropout_p,
                   **kwargs)
def resnet34(pretrained=False, progress=True, dropout_p=0.2,**kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,dropout_p,
                   **kwargs)

def resnet50(pretrained=False, progress=True, dropout_p=0.2,**kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,dropout_p,
                   **kwargs)

def resnet101(pretrained=False, progress=True, dropout_p=0.2,**kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,dropout_p,
                   **kwargs)

def resnet152(pretrained=False, progress=True, dropout_p=0.2,**kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,dropout_p,
                   **kwargs)

