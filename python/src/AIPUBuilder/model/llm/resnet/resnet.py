import torch
from AIPUBuilder.core import*  # noqa
from AIPUBuilder.logger import INFO, ERROR, WARN, DEBUG
from AIPUBuilder import ops
from AIPUBuilder import nn

from AIPUBuilder.model.basemodel import TCVisionHub, BaseModel
from AIPUBuilder.model.register import register_torchvision_hub, register_nn
from AIPUBuilder.model.utils import update_tensor_data


@register_torchvision_hub(['resnet50'])
class ResNet50Hub(TCVisionHub):
    def __init__(self, config=None, m=None):
        if m is None:
            import torchvision.models as models
            m = models
        self.model = m.resnet50(weights="IMAGENET1K_V2")


@register_torchvision_hub(['resnet18'])
class ResNet18Hub(TCVisionHub):
    def __init__(self, config=None, m=None):
        if m is None:
            import torchvision.models as models
            m = models
        self.model = m.resnet18(weights="IMAGENET1K_V1")


def fuse_conv_bn(cw, cb, bw, bb, brm, brv, eps=1e-05):
    cw = cw['data']
    cb = cb.get('data', torch.tensor(0.0).to(cw.device))
    bw = bw['data']
    bb = bb['data']
    brm = brm['data']
    brv = brv['data']

    factor = bw / torch.sqrt(brv + torch.tensor(eps).to(brv))
    w_fused = cw * factor.reshape(-1, 1, 1, 1)
    b_fused = (cb - brm) * factor + bb
    return w_fused, b_fused


class Conv1x1Or3x3(nn.Module):

    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        prefix,
        conv_bn_idx,
        kernels,
        conv_name='conv',
        bn_name='bn',
        strides=[1, 1],
        dilation=[1, 1],
        padding=[0, 0, 0, 0],
        with_bn=True,
        with_activation='None',
    ):
        """3x3 convolution with padding, 1x1 without padding"""
        # prefix = f'layer{layer_id}.{block_id}.'
        if kernels[0] == 3:
            padding = dilation + dilation

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernels=kernels,
            strides=strides,
            dilations=dilation,
            padding=padding,
            activation=with_activation,
            name=f"{prefix}_conv",
        )
        self.weights = config.get("weights")
        conv_w = self.weights.get(f'{prefix}{conv_name}{conv_bn_idx}.weight')
        conv_b = self.weights.get(f'{prefix}{conv_name}{conv_bn_idx}.bias')
        fused_w, fused_b = conv_w, conv_b
        if with_bn:
            bn_w = self.weights.get(f'{prefix}{bn_name}{conv_bn_idx}.weight')
            bn_b = self.weights.get(f'{prefix}{bn_name}{conv_bn_idx}.bias')
            bn_rm = self.weights.get(f'{prefix}{bn_name}{conv_bn_idx}.running_mean')
            bn_rv = self.weights.get(f'{prefix}{bn_name}{conv_bn_idx}.running_var')

            # fuse_conv_bn
            fused_w, fused_b = fuse_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_rm, bn_rv)
            fused_w = fused_w.permute(0, 2, 3, 1)
        else:
            fused_w['data'] = conv_w['data'].permute(0, 2, 3, 1)

        update_tensor_data(self.conv.weights, fused_w)
        update_tensor_data(self.conv.biases, fused_b)

    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        config,
        inplanes,
        planes,
        layer_id,
        block_id,
        strides=[1, 1],
        dilation=[1, 1],
        downsample=None,
        groups=1,
        base_width=64,
        norm_layer='bn',
        prefix=''
    ):
        if norm_layer != 'bn':
            WARN(f"now only support norm_layer is batchnorm")

        if groups != 1 or base_width != 64:
            ERROR(f"BasicBlock only supports groups=1 and base_widht=64")
         # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        prefix = f"{prefix}layer{layer_id}.{block_id}."
        INFO(f"begin to init {prefix}")

        self.conv1 = Conv1x1Or3x3(
            config,
            in_channels=inplanes,
            out_channels=planes,
            prefix=prefix,
            conv_bn_idx=1,
            kernels=[3, 3],
            strides=strides,
            with_activation='relu'
        )

        self.conv2 = Conv1x1Or3x3(
            config,
            in_channels=planes,
            out_channels=planes,
            prefix=prefix,
            conv_bn_idx=2,
            kernels=[3, 3],
        )

        self.downsample_module = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample_module is not None:
            residual = self.downsample_module(x)
        out = ops.add(out, residual, activation='relu')
        return out


class BottleNeck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        config,
        inplanes,
        planes,
        layer_id,
        block_id,
        strides=[1, 1],
        dilation=[1, 1],
        downsample=None,
        groups=1,
        base_width=64,
        norm_layer='bn',
        prefix=''
    ):
        if norm_layer != 'bn':
            WARN(f"now only support norm_layer is batchnorm")
        width = int(planes * (base_width/64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        prefix = f"{prefix}layer{layer_id}.{block_id}."
        INFO(f"init {config.get('backbone', config.get('model_name', 'resnet'))} {prefix}")
        self.conv1 = Conv1x1Or3x3(
            config,
            in_channels=inplanes,
            out_channels=width,
            prefix=prefix,
            conv_bn_idx=1,
            kernels=[1, 1],
            with_activation='relu'
        )

        self.conv2 = Conv1x1Or3x3(
            config,
            in_channels=width,
            out_channels=width,
            prefix=prefix,
            conv_bn_idx=2,
            kernels=[3, 3],
            strides=strides,
            dilation=dilation,
            padding=dilation + dilation,
            with_activation='relu'
        )

        self.conv3 = Conv1x1Or3x3(
            config,
            in_channels=width,
            out_channels=planes*self.expansion,
            prefix=prefix,
            conv_bn_idx=3,
            kernels=[1, 1],
        )

        self.downsample_module = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample_module is not None:
            residual = self.downsample_module(x)
        out = ops.add(out, residual, activation='relu')
        return out


class ResNetBackBoneModel(nn.Module):
    # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L334
    def __init__(self, config, weights, prefix=''):
        self.config = config
        self.weights = weights
        self.prefix = prefix
        self.model_name = config.get("backbone", config.get('model_name', 'resnet50'))
        self.model_basic_configs = {
            'resnet50': {'block_type': BottleNeck, 'layers': [3, 4, 6, 3]},
            'resnet34': {'block_type': BasicBlock, 'layers': [3, 4, 6, 3]},
            'resnet18': {'block_type': BasicBlock, 'layers': [2, 2, 2, 2]},
        }
        self.basic_config = self.model_basic_configs.get(self.model_name)
        self.groups = config.get('groups', 1)
        self.base_width = config.get('width_per_group', 64)
        self.norm_layer = config.get('norm_layer', 'bn')
        self.inplanes = 64
        self.dilation = 1
        layers = self.basic_config.get('layers')
        block_type = self.basic_config.get('block_type')
        self.config.update({'resnet_block_type': block_type})

        replace_stride_with_dilation = config.get('replace_stride_with_dilation', None)
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if self.norm_layer != 'bn':
            WARN(f"now only support norm_layer is batchnorm")

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.inplanes,
            kernels=[7, 7],
            strides=[2, 2],
            padding=[3, 3, 3, 3],
            activation='relu'
        )

        conv1_w = self.weights.get(f'{prefix}conv1.weight')
        conv1_b = self.weights.get(f'{prefix}conv1.bias')
        bn1_w = self.weights.get(f'{prefix}bn1.weight')
        bn1_b = self.weights.get(f'{prefix}bn1.bias')
        bn1_rm = self.weights.get(f'{prefix}bn1.running_mean')
        bn1_rv = self.weights.get(f'{prefix}bn1.running_var')

        # fuse_conv_bn
        fused_w, fused_b = fuse_conv_bn(conv1_w, conv1_b, bn1_w, bn1_b, bn1_rm, bn1_rv)
        update_tensor_data(self.conv1.weights, fused_w.permute(0, 2, 3, 1))
        update_tensor_data(self.conv1.biases, fused_b)

        self.max_pool = nn.MaxPool(kernels=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1])
        self.layer1 = self._make_layer(block_type, planes=64, block_num=layers[0], stride=1, dilate=False, layer_id=1)
        self.layer2 = self._make_layer(
            block_type, planes=128, block_num=layers[1], stride=2, dilate=replace_stride_with_dilation[0], layer_id=2)
        self.layer3 = self._make_layer(
            block_type, planes=256, block_num=layers[2], stride=2, dilate=replace_stride_with_dilation[1], layer_id=3)
        self.layer4 = self._make_layer(
            block_type, planes=512, block_num=layers[3], stride=2, dilate=replace_stride_with_dilation[2], layer_id=4)

    def _make_layer(
        self,
        block,
        planes,
        block_num,
        stride,
        dilate,
        layer_id,
    ):
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Conv1x1Or3x3(
                self.config,
                in_channels=self.inplanes,
                out_channels=planes * block.expansion,
                prefix=f'{self.prefix}layer{layer_id}.0.',
                conv_bn_idx="",
                conv_name='downsample.0',
                bn_name="downsample.1",
                kernels=[1, 1],
                strides=[stride, stride],
            )

        layers = []
        layers.append(
            block(
                self.config,
                self.inplanes,
                planes,
                layer_id=layer_id,
                block_id=0,
                strides=[stride, stride],
                dilation=[previous_dilation, previous_dilation],
                downsample=downsample,
                base_width=self.base_width,
                prefix=self.prefix
            )
        )
        self.inplanes = planes * block.expansion
        for blk_idx in range(1, block_num):
            layers.append(
                block(
                    self.config,
                    self.inplanes,
                    planes,
                    layer_id=layer_id,
                    block_id=blk_idx,
                    # strides=[stride, stride],
                    dilation=[self.dilation, self.dilation],
                    base_width=self.base_width,
                    prefix=self.prefix
                )
            )
        return layers

    def forward(self, x):
        conv1_out = self.conv1(x)
        out = self.max_pool(conv1_out)
        for l in self.layer1 + self.layer2 + self.layer3 + self.layer4:
            out = l(out)
        return out


class ResNet(nn.Module):

    def __init__(self, config, weights):
        self.config = config
        self.backbone = ResNetBackBoneModel(config, weights)
        num_classes = config.get('num_classes', 1000)
        block_type = config.get('resnet_block_type', None)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=[1, 1])
        self.fc = nn.Linear(in_features=512 * block_type.expansion, out_features=num_classes)
        fc_w = weights.get("fc.weight")
        fc_b = weights.get('fc.bias')
        update_tensor_data(self.fc.weights, fc_w)
        update_tensor_data(self.fc.biases, fc_b)

    def forward(self, x):
        backbone_out = self.backbone(x)
        avg_pool = self.avg_pool(backbone_out)
        fc_in = avg_pool.reshape([-1, self.fc.in_features])
        fc_out = self.fc(fc_in)
        return fc_out


@register_nn(['resnet50', 'resnet18', 'resnet34'])
class ResNetNNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.model = ResNet(config, weights)
        self.aipu_g = None

    def forward(self):
        batch = self.config.get('batch', 1)
        height = self.config.get('image_height', 224)
        width = self.config.get('image_width', 224)
        channel = self.config.get('image_channel', 3)
        with Graph('resnet') as self.aipu_g:
            image = ops.input(
                [batch, height, width, channel],
                Dtype.FP32,
                name='input',
                exprs=['batch', 'height', 'width', 'channel'])
            self.model(image)
        return self.aipu_g
