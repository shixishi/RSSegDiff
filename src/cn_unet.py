# 分割网络 保证CN和NN 结构一致

import math
import numpy as np
import torch as th
import torch.nn as nn
from .nn import zero_module
from typing import Tuple, List
import torch.nn.functional as F


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock2D(nn.Module):
    """ResBlock
    Args:
        channels: input channels
         dropout: the rate of dropout. 
        out_channels: if specified, the number of out channels
        use_conv: if True and out_channels is specified, use a spatial
            convolution instead of a smaller 1x1 convolution to change the
            channels in the skip connection.
        up: if True, use this block for upsampling.
        down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,   
        dropout,    
        out_channels=None, 
        use_conv=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        # self.updown = up or down

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1)
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            # 使用 3 * 3 padding为1 的卷积，保证空间尺寸不变，但改变通道数 
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            # 使用 1 * 1 的卷积，改变通道数
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            # 断言channels必须能整除 num_head_channels
            # num_head_channels 是每个head的维度
            # channels 是输入特征图的通道数
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            # 动态计算num_heads
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(32, channels)   # GroupNorm
        self.qkv = nn.Conv1d(channels, channels * 3, 1)  # 使用卷积方式进行QKV投影计算
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        # 输入的x是NxCxHxW
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)   # reshape为NxCx(HW)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(ch)
        # scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        # weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = th.softmax(weight, dim=-1)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        # scale = 1 / math.sqrt(math.sqrt(ch))
        scale = 1 / math.sqrt(ch)
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        # weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = th.softmax(weight, dim=-1)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# ---------------------------------------------------------------------
# Condition UNet ———————————————————————————
# ---------------------------------------------------------------------
class ConditionUNet(nn.Module):
    """UNet encoder–decoder that produces segmentation logits **and** its
    bottleneck feature (1/32 scale) for fusion.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=False,
        use_new_attention_order=False
    ):
        super().__init__()

        # 上采样阶段的注意力头数量
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        # self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels  # 就是num_classes
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample


        # self.num_classes = num_classes
        # input_ch 记录初始卷积层变换后的通道数
        # 最后一个 输出卷积层，其输入通道数是input_ch, 输出通道数是num_classes
        # ch 记录每个层的输入通道数，保持变化的
        ch = input_ch = int(channel_mult[0] * model_channels)
        # ---- Encoder ------------------
        # 初始卷积层，负责通道变换，空间尺寸不变
        self.input_blocks = nn.ModuleList(
            nn.Sequential(nn.Conv2d(in_channels, ch, 3, padding=1))
        )
        self._feature_size = ch
        input_block_chans = [ch]   # 将下采样层的输出通道数记录下来
        ds = 1  # current downsample ratio
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # 每个
                layers = [ResBlock2D(ch, dropout, out_channels=int(mult * model_channels))]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:   # ds是下采样比率
                    layers.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # 最后一个层不需要下采样

                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock2D(ch, dropout, out_channels=out_ch, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2   # 下采样比例 * 2
                self._feature_size += ch

        # ---- mid_layer ------------
        self.middle_block = nn.Sequential(
            ResBlock2D(ch, dropout),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order),
            ResBlock2D(ch, dropout),
        )
        self._feature_size += ch

        # ---- Decoder -------------------
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock2D(ch + ich, dropout, out_channels=int(model_channels * mult))]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads_upsample, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock2D(ch, dropout, out_channels=out_ch, up=True)
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        # ---- Head -----------
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1))
        )

    def forward(self, x):
        hs = []
        h = x
        for blk in self.input_blocks:
            h = blk(h)
            hs.append(h)
        h_mid = self.middle_block(h)
        h = h_mid
        for blk in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = blk(h)
        logits = self.out(h)
        return logits, h_mid