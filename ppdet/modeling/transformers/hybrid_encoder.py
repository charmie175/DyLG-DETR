import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv
from ..backbones.cspresnet import RepVggBlock
from ppdet.modeling.transformers.detr_transformer import TransformerEncoder
from ..initializer import xavier_uniform_, linear_init_
from ..layers import MultiHeadAttention
from paddle import ParamAttr
from paddle.regularizer import L2Decay

__all__ = ['HybridEncoder']


class CSPRepLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv4 = SDM(
            hidden_channels, hidden_channels,stride=1)

        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(
                hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)

        else:
            self.conv3 = nn.Identity() 

    def forward(self, x):

        def _check(name, t):
            if bool(paddle.all(paddle.isfinite(t)).numpy()):
                return

            nan_num = int(
                paddle.isnan(t).astype('int64').sum().numpy())
            inf_num = int(
                paddle.isinf(t).astype('int64').sum().numpy())

            print("\n" + "=" * 100)
            print("[CSPRepLayer FIRST NON-FINITE]")
            print("stage :", name)
            print("shape :", list(t.shape))
            print("bad   :", nan_num + inf_num, "/", int(t.numel()))
            print("nan   :", nan_num)
            print("inf   :", inf_num)
            print("=" * 100 + "\n")

            raise FloatingPointError(
                "Non-finite inside CSPRepLayer: {}".format(name))

        _check("INPUT", x)

        # branch 1
        x_1 = self.conv1(x)
        _check("conv1.output", x_1)

        # SDM branch
        x_3 = self.conv4(x_1)
        _check("SDM.output", x_3)

        # RepVGG branch
        x_1 = self.bottlenecks(x_1)
        _check("bottlenecks.output", x_1)

        # branch 2
        x_2 = self.conv2(x)
        _check("conv2.output", x_2)

        # residual fusion
        fused = x_1 + x_2 + x_3
        _check("fusion.output", fused)

        out = self.conv3(fused)
        _check("conv3.output", out)

        return out
    
class ECC(nn.Layer):
    def __init__(self, in_chn, out_chn, stride=1, groups=1, dilation=1):
        super(ECC, self).__init__()
        self.in_channels = in_chn
        self.out_channels = out_chn
        self.kernel_size = 3
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = self.dilation

        # 可学习的方向核
        self.direction_kernels = paddle.create_parameter(
            shape=[8, 1, 3, 3],
            dtype='float32'
        )
        # 初始化为8个方向的边缘检测核
        self._init_direction_kernels()
        
        # 方向注意力
        self.direction_attention = nn.Sequential(
            nn.Conv2D(8 * in_chn, in_chn, 1),
            nn.BatchNorm2D(in_chn),
            nn.ReLU(),
            nn.Conv2D(in_chn, 8, 1),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2D(in_chn, out_chn, 1),
            nn.BatchNorm2D(out_chn)
        )

    def _init_direction_kernels(self):
        # 初始化8个方向的边缘检测核作为起点
        init_kernels = paddle.to_tensor([
            [[ 1.,  0., -1.], [ 2.,  0., -2.], [ 1.,  0., -1.]], # 水平
            [[ 1.,  2.,  1.], [ 0.,  0.,  0.], [-1., -2., -1.]], # 垂直
            [[ 2.,  1.,  0.], [ 1.,  0., -1.], [ 0., -1., -2.]], # 45度
            [[ 0.,  1.,  2.], [-1.,  0.,  1.], [-2., -1.,  0.]], # 135度
            [[-2., -1.,  0.], [-1.,  0.,  1.], [ 0.,  1.,  2.]], # 225度
            [[ 0., -1., -2.], [ 1.,  0., -1.], [ 2.,  1.,  0.]], # 315度
            [[ 1., -2.,  1.], [-2.,  4., -2.], [ 1., -2.,  1.]], # 拉普拉斯
            [[-1., -1., -1.], [-1.,  8., -1.], [-1., -1., -1.]]  # LOG
        ], dtype='float32')
        init_kernels = paddle.reshape(init_kernels, [8, 1, 3, 3])
        self.direction_kernels.set_value(init_kernels)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x_reshaped = x.reshape([B*C, 1, H, W])
        
        # 应用方向核

        edge_maps = F.conv2d(
        x_reshaped,
        self.direction_kernels,  # 核形状 [8, 1, 3, 3]
        padding=self.padding,
        groups=1               # 每个样本每个通道独立处理
    ) 
        edge_maps = edge_maps.reshape([B, C, 8, H, W]).transpose([0, 2, 1, 3, 4])
        
        # 计算方向注意力
        attention_input = edge_maps.reshape([B, 8*C, H, W])
        attention = self.direction_attention(attention_input)  # 输出应为 [B, 8, H, W]
    
    # 调整注意力形状为 [B, 8, 1, H, W] 以便广播
        attention = attention.unsqueeze(2)  # 在位置2插入维度1
        edge_maps = edge_maps * attention
        edge_maps = edge_maps.sum(axis=1)
        
        # 输出投影
        out = self.out_proj(edge_maps)
        return out


class LargeKernelDecomp(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # 水平分解
        self.h1 = nn.Conv2D(
            in_channels, out_channels, (1, kernel_size),
            stride=1, padding=(0, padding))
        self.h2 = nn.Conv2D(
            out_channels, out_channels, (kernel_size, 1),
            stride=stride, padding=(padding, 0))
        self.h_bn = nn.BatchNorm2D(out_channels)
        
        # 垂直分解
        self.v1 = nn.Conv2D(
            in_channels, out_channels, (kernel_size, 1),
            stride=1, padding=(padding, 0))
        self.v2 = nn.Conv2D(
            out_channels, out_channels, (1, kernel_size),
            stride=stride, padding=(0, padding))
        self.v_bn = nn.BatchNorm2D(out_channels)
        
        # 注意力融合
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(out_channels * 2, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2D(out_channels // 8, out_channels * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        def _check(name, t):
            if bool(paddle.all(paddle.isfinite(t)).numpy()):
                return

            nan_num = int(
                paddle.isnan(t).astype('int64').sum().numpy())
            inf_num = int(
                paddle.isinf(t).astype('int64').sum().numpy())

            print("\n" + "=" * 100)
            print("[LargeKernelDecomp FIRST NON-FINITE]")
            print("kernel:", self.kernel_size)
            print("stage :", name)
            print("shape :", list(t.shape))
            print("bad   :", nan_num + inf_num, "/", int(t.numel()))
            print("nan   :", nan_num)
            print("inf   :", inf_num)
            print("=" * 100 + "\n")

            raise FloatingPointError(
                "Non-finite LargeKernelDecomp K{}: {}".format(
                    self.kernel_size, name))

        _check("INPUT", x)

        # horizontal path
        h = self.h1(x)
        _check("h1.output", h)

        h = self.h2(h)
        _check("h2.output", h)

        h = self.h_bn(h)
        _check("h_bn.output", h)

        # vertical path
        v = self.v1(x)
        _check("v1.output", v)

        v = self.v2(v)
        _check("v2.output", v)

        v = self.v_bn(v)
        _check("v_bn.output", v)

        # attention
        feat = paddle.concat([h, v], axis=1)
        _check("concat.output", feat)

        attn = self.attn(feat)
        _check("attention.output", attn)

        h_attn, v_attn = paddle.split(
            attn, 2, axis=1)

        _check("h_attention", h_attn)
        _check("v_attention", v_attn)

        out = h * h_attn + v * v_attn
        _check("FINAL.output", out)

        return out


class SDM(nn.Layer):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.out_channels = out_channels
        
        # 多尺度分解大核
        self.branch1 = LargeKernelDecomp(in_channels, out_channels//3, kernel_size=13, stride=1)
        self.branch2 = LargeKernelDecomp(in_channels, out_channels//3, kernel_size=9, stride=1)
        self.branch3 = LargeKernelDecomp(in_channels, out_channels - 2 * (out_channels//3), kernel_size=7, stride=1)
        
        # 特征融合
        self.fusion = BaseConv(
            in_channels=out_channels,
            out_channels=out_channels,
            ksize=1,
            stride=stride)
        
    def forward(self, x):

        def _check(name, t):
            if bool(paddle.all(paddle.isfinite(t)).numpy()):
                return

            nan_num = int(
                paddle.isnan(t).astype('int64').sum().numpy())
            inf_num = int(
                paddle.isinf(t).astype('int64').sum().numpy())

            print("\n" + "=" * 100)
            print("[SDM FIRST NON-FINITE]")
            print("stage :", name)
            print("shape :", list(t.shape))
            print("bad   :", nan_num + inf_num, "/", int(t.numel()))
            print("nan   :", nan_num)
            print("inf   :", inf_num)
            print("=" * 100 + "\n")

            raise FloatingPointError(
                "Non-finite inside SDM: {}".format(name))

        _check("INPUT", x)

        # K=13
        x1 = self.branch1(x)
        _check("branch1.K13.output", x1)

        # K=9
        x2 = self.branch2(x)
        _check("branch2.K9.output", x2)

        # K=7
        x3 = self.branch3(x)
        _check("branch3.K7.output", x3)

        # concatenate three MRFD branches
        out = paddle.concat(
            [x1, x2, x3],
            axis=1)

        _check("concat.output", out)

        # fusion
        out = self.fusion(out)

        _check("fusion.output", out)

        return out




@register
class TransformerLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

@register
@serializable
class HybridEncoder(nn.Layer):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.LayerList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channel, hidden_dim, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(
                        hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))
        # encoder transformer
        self.encoder = nn.LayerList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act)
                    )
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self, feats, for_mot=False):

        # ============================================================
        # DyLG NaN DEBUG
        # Locate the FIRST non-finite stage inside HybridEncoder.
        # ============================================================
        def _check_finite(name, x):
            if not isinstance(x, paddle.Tensor):
                return

            if bool(paddle.all(paddle.isfinite(x)).numpy()):
                return

            nan_num = int(
                paddle.isnan(x).astype('int64').sum().numpy())
            inf_num = int(
                paddle.isinf(x).astype('int64').sum().numpy())
            total = int(x.numel())

            print("\n" + "=" * 100)
            print("[NECK FIRST NON-FINITE]")
            print("stage :", name)
            print("shape :", list(x.shape))
            print("dtype :", x.dtype)
            print("bad   :", nan_num + inf_num, "/", total)
            print("nan   :", nan_num)
            print("inf   :", inf_num)

            finite_mask = paddle.isfinite(x)
            if bool(paddle.any(finite_mask).numpy()):
                v = paddle.masked_select(x, finite_mask)
                print("finite min  :", float(v.min().numpy()))
                print("finite max  :", float(v.max().numpy()))
                print("finite mean :", float(v.mean().numpy()))

            print("=" * 100 + "\n")

            raise FloatingPointError(
                "Non-finite inside HybridEncoder: {}".format(name))

        assert len(feats) == len(self.in_channels)

        # ------------------------------------------------------------
        # 0. Backbone -> Neck
        # ------------------------------------------------------------
        for i, feat in enumerate(feats):
            _check_finite(
                "backbone_feat[{}]".format(i),
                feat)

        # ------------------------------------------------------------
        # 1. Input projection
        # ------------------------------------------------------------
        proj_feats = []

        for i, feat in enumerate(feats):
            x = self.input_proj[i](feat)

            _check_finite(
                "input_proj[{}].output".format(i),
                x)

            proj_feats.append(x)

        # ------------------------------------------------------------
        # 2. Transformer encoder
        # ------------------------------------------------------------
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):

                h, w = proj_feats[enc_ind].shape[2:]

                _check_finite(
                    "encoder[{}].input".format(i),
                    proj_feats[enc_ind])

                src_flatten = (
                    proj_feats[enc_ind]
                    .flatten(2)
                    .transpose([0, 2, 1])
                )

                _check_finite(
                    "encoder[{}].src_flatten".format(i),
                    src_flatten)

                if self.training or self.eval_size is None:
                    pos_embed = (
                        self.build_2d_sincos_position_embedding(
                            w,
                            h,
                            self.hidden_dim,
                            self.pe_temperature
                        )
                    )
                else:
                    pos_embed = getattr(
                        self,
                        f'pos_embed{enc_ind}',
                        None
                    )

                if pos_embed is not None:
                    _check_finite(
                        "encoder[{}].pos_embed".format(i),
                        pos_embed)

                memory = self.encoder[i](
                    src_flatten,
                    pos_embed=pos_embed
                )

                _check_finite(
                    "encoder[{}].memory".format(i),
                    memory)

                proj_feats[enc_ind] = (
                    memory
                    .transpose([0, 2, 1])
                    .reshape(
                        [-1, self.hidden_dim, h, w]
                    )
                )

                _check_finite(
                    "encoder[{}].output".format(i),
                    proj_feats[enc_ind])

        # ------------------------------------------------------------
        # 3. Top-down FPN
        # ------------------------------------------------------------
        inner_outs = [proj_feats[-1]]

        _check_finite(
            "fpn.initial",
            inner_outs[0])

        for idx in range(
                len(self.in_channels) - 1,
                0,
                -1):

            block_idx = (
                len(self.in_channels) - 1 - idx
            )

            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            _check_finite(
                "fpn[{}].feat_high.input".format(
                    block_idx),
                feat_heigh)

            _check_finite(
                "fpn[{}].feat_low.input".format(
                    block_idx),
                feat_low)

            # --------------------------------------------------------
            # DEBUG lateral BaseConv: input -> parameters -> buffers -> output
            # --------------------------------------------------------
            lateral = self.lateral_convs[block_idx]

            _check_finite(
                "lateral_convs[{}].input".format(block_idx),
                feat_heigh
            )

            # Check all trainable parameters
            for param_name, param in lateral.named_parameters():
                _check_finite(
                    "lateral_convs[{}].param.{}".format(
                        block_idx, param_name),
                    param
                )

            # Check BN running statistics / buffers
            for buffer_name, buffer in lateral.named_buffers():
                if isinstance(buffer, paddle.Tensor):
                    _check_finite(
                        "lateral_convs[{}].buffer.{}".format(
                            block_idx, buffer_name),
                        buffer
                    )

            # DEBUG: split BaseConv into Conv -> BN -> SiLU
            feat_heigh = lateral.conv(feat_heigh)
            _check_finite(
                "lateral_convs[{}].conv.output".format(block_idx),
                feat_heigh
            )

            # ============================================================
            # FULL BN DIAGNOSTIC PROBE
            # ============================================================
            bn_input = feat_heigh
            bn = lateral.bn

            # Save statistics before BN forward
            batch_mean = paddle.mean(bn_input, axis=[0, 2, 3])
            batch_var = paddle.mean(
                paddle.square(
                    bn_input - batch_mean.reshape([1, -1, 1, 1])
                ),
                axis=[0, 2, 3]
            )

            print(
                "[BN STATE] block={} training={} "
                "use_global_stats={} epsilon={}".format(
                    block_idx,
                    bn.training,
                    getattr(bn, "_use_global_stats", "NA"),
                    getattr(bn, "_epsilon", "NA")
                )
            )

            _check_finite(
                "lateral_convs[{}].bn.input".format(block_idx),
                bn_input
            )
            _check_finite(
                "lateral_convs[{}].bn.batch_mean".format(block_idx),
                batch_mean
            )
            _check_finite(
                "lateral_convs[{}].bn.batch_var".format(block_idx),
                batch_var
            )
            _check_finite(
                "lateral_convs[{}].bn.weight".format(block_idx),
                bn.weight
            )
            _check_finite(
                "lateral_convs[{}].bn.bias".format(block_idx),
                bn.bias
            )
            _check_finite(
                "lateral_convs[{}].bn.running_mean".format(block_idx),
                bn._mean
            )
            _check_finite(
                "lateral_convs[{}].bn.running_var".format(block_idx),
                bn._variance
            )

            print(
                "[BN STATS] block={} "
                "batch_var_min={} batch_var_max={} "
                "running_var_min={} running_var_max={}".format(
                    block_idx,
                    float(batch_var.min().numpy()),
                    float(batch_var.max().numpy()),
                    float(bn._variance.min().numpy()),
                    float(bn._variance.max().numpy())
                )
            )

            # Real Paddle BN forward
            bn_output = bn(bn_input)

            finite_mask = paddle.isfinite(bn_output)
            if not bool(paddle.all(finite_mask).numpy()):

                # Find exactly which channels contain NaN/Inf
                bad_per_channel = paddle.sum(
                    (~finite_mask).astype("int32"),
                    axis=[0, 2, 3]
                )

                bad_channels = paddle.nonzero(
                    bad_per_channel > 0
                ).flatten().numpy().tolist()

                print("\n" + "=" * 100)
                print("[BN FATAL] block={}".format(block_idx))
                print("[BN FATAL] bad_channels={}".format(bad_channels))

                eps = float(getattr(bn, "_epsilon", 1e-5))

                for c in bad_channels:
                    c = int(c)

                    x_c = bn_input[:, c, :, :]

                    gamma = float(bn.weight[c].numpy())
                    beta = float(bn.bias[c].numpy())
                    run_mean = float(bn._mean[c].numpy())
                    run_var = float(bn._variance[c].numpy())
                    b_mean = float(batch_mean[c].numpy())
                    b_var = float(batch_var[c].numpy())

                    print("-" * 100)
                    print("[BAD CHANNEL] c={}".format(c))
                    print(
                        "input: min={} max={} mean={}".format(
                            float(x_c.min().numpy()),
                            float(x_c.max().numpy()),
                            float(x_c.mean().numpy())
                        )
                    )
                    print("gamma(weight)={}".format(gamma))
                    print("beta(bias)={}".format(beta))
                    print("running_mean={}".format(run_mean))
                    print("running_var={}".format(run_var))
                    print("batch_mean={}".format(b_mean))
                    print("batch_var={}".format(b_var))
                    print("epsilon={}".format(eps))
                    print("running_var + eps={}".format(run_var + eps))
                    print("batch_var + eps={}".format(b_var + eps))

                    # Manual BN using running statistics
                    manual_running = (
                        (x_c - run_mean) /
                        paddle.sqrt(
                            paddle.to_tensor(
                                run_var + eps,
                                dtype=x_c.dtype
                            )
                        )
                    ) * gamma + beta

                    manual_running_bad = int(
                        paddle.sum(
                            (~paddle.isfinite(manual_running)).astype("int32")
                        ).numpy()
                    )

                    # Manual BN using current batch statistics
                    manual_batch = (
                        (x_c - b_mean) /
                        paddle.sqrt(
                            paddle.to_tensor(
                                b_var + eps,
                                dtype=x_c.dtype
                            )
                        )
                    ) * gamma + beta

                    manual_batch_bad = int(
                        paddle.sum(
                            (~paddle.isfinite(manual_batch)).astype("int32")
                        ).numpy()
                    )

                    print(
                        "manual_running_bad={}".format(
                            manual_running_bad
                        )
                    )
                    print(
                        "manual_batch_bad={}".format(
                            manual_batch_bad
                        )
                    )

                print("=" * 100 + "\n")

                raise FloatingPointError(
                    "BN produced non-finite values at "
                    "lateral_convs[{}], channels={}".format(
                        block_idx, bad_channels
                    )
                )

            feat_heigh = bn_output

            _check_finite(
                "lateral_convs[{}].bn.output".format(block_idx),
                feat_heigh
            )

            sigmoid_out = F.sigmoid(feat_heigh)
            _check_finite(
                "lateral_convs[{}].sigmoid.output".format(block_idx),
                sigmoid_out
            )

            feat_heigh = feat_heigh * sigmoid_out
            _check_finite(
                "lateral_convs[{}].silu.output".format(block_idx),
                feat_heigh
            )

            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh,
                scale_factor=2.,
                mode="nearest"
            )

            _check_finite(
                "fpn[{}].upsample".format(
                    block_idx),
                upsample_feat)

            cat_feat = paddle.concat(
                [upsample_feat, feat_low],
                axis=1
            )

            _check_finite(
                "fpn_blocks[{}].input".format(
                    block_idx),
                cat_feat)

            inner_out = (
                self.fpn_blocks[block_idx](
                    cat_feat)
            )

            _check_finite(
                "fpn_blocks[{}].output".format(
                    block_idx),
                inner_out)

            inner_outs.insert(
                0,
                inner_out
            )

        # ------------------------------------------------------------
        # 4. Bottom-up PAN
        # ------------------------------------------------------------
        outs = [inner_outs[0]]

        _check_finite(
            "pan.initial",
            outs[0])

        for idx in range(
                len(self.in_channels) - 1):

            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]

            _check_finite(
                "pan[{}].feat_low.input".format(idx),
                feat_low)

            _check_finite(
                "pan[{}].feat_height.input".format(idx),
                feat_height)

            downsample_feat = (
                self.downsample_convs[idx](
                    feat_low)
            )

            _check_finite(
                "downsample_convs[{}].output".format(
                    idx),
                downsample_feat)

            cat_feat = paddle.concat(
                [downsample_feat, feat_height],
                axis=1
            )

            _check_finite(
                "pan_blocks[{}].input".format(idx),
                cat_feat)

            out = self.pan_blocks[idx](
                cat_feat)

            _check_finite(
                "pan_blocks[{}].output".format(idx),
                out)

            outs.append(out)

        # ------------------------------------------------------------
        # 5. Final neck outputs
        # ------------------------------------------------------------
        for i, out in enumerate(outs):
            _check_finite(
                "neck.final_output[{}]".format(i),
                out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]
