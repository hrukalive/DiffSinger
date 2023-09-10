import math
from typing import Optional

import torch
import torch.nn as nn

import torch.nn.functional as F


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


class RelativeFFTBlock(nn.Module):
    """ FFT Block with Relative Multi-Head Attention """

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.,
                 window_size=None, block_length=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(RelativeSelfAttention(hidden_channels, hidden_channels, n_heads,
                                                          window_size=window_size, p_dropout=p_dropout,
                                                          block_length=block_length))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(
                hidden_channels, hidden_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask=None):

        if x_mask is not None:
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        else:
            attn_mask = None

        for i in range(self.n_layers):
            if x_mask is not None:
                x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        if x_mask is not None:
            x = x * x_mask
        return x


class RelativeSelfAttention(nn.Module):
    """ Relative Multi-Head Attention """

    def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share=True, p_dropout=0.,
                 block_length=None, proximal_bias=False, proximal_init=False):
        super(RelativeSelfAttention, self).__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(
                n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(
                n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels,
                           t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels,
                           t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(self.k_channels)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(
                rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + \
                     self._attention_bias_proximal(t_s).to(
                         device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                block_mask = torch.ones_like(
                    scores).triu(-self.block_length).tril(self.block_length)
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s)
            output = output + \
                     self._matmul_with_relative_values(
                         relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(
            b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
                                   slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape(
            [[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view(
            [batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, convert_pad_shape(
            [[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """
        Bias for self-attention to encourage attention to close positions.
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p_dropout=0., activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask=None):
        if x_mask is not None:
            x = self.conv(x * x_mask)
        else:
            x = self.conv(x)

        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


Conv1dModel = nn.Conv1d  # 有毒 删


class Depthwise_Separable_Conv1D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            padding_mode='zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ):
        super().__init__()
        self.depth_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                    groups=in_channels, stride=stride, padding=padding, dilation=dilation, bias=bias,
                                    padding_mode=padding_mode, device=device, dtype=dtype)
        self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias,
                                    device=device, dtype=dtype)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))


def set_Conv1dModel(use_depthwise_conv):
    global Conv1dModel
    Conv1dModel = Depthwise_Separable_Conv1D if use_depthwise_conv else nn.Conv1d


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


@torch.jit.script
def add_and_GRU(input_a, input_b):
    in_act = input_a + input_b
    x1, x2 = in_act.chunk(2, dim=1)
    t_act = torch.tanh(x2)
    s_act = torch.sigmoid(x1)
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels  # condition用的
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        self.condition_layers = torch.nn.ModuleList()

        # if gin_channels != 0:
        #     cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
        #     # self.cond_layer = weight_norm_modules(cond_layer, name='weight')
        #     self.cond_layer=cond_layer

        for i in range(n_layers):

            if gin_channels != 0:
                cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels, 1)
                # self.cond_layer = weight_norm_modules(cond_layer, name='weight')
                # self.cond_layer = cond_layer
            else:
                cond_layer = nn.Identity()
            self.condition_layers.append(cond_layer)

            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = Conv1dModel(hidden_channels, 2 * hidden_channels, kernel_size,
                                   dilation=dilation, padding=padding)
            # in_layer = weight_norm_modules(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            # res_skip_layer = weight_norm_modules(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask=None, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        # if g is not None:
        #     g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:

                condition = self.condition_layers[i](g)
            else:
                condition = torch.zeros_like(x_in)

            # acts = fused_add_tanh_sigmoid_multiply(  # GRU 这不就是wavnet的那个 GRU
            #     x_in,
            #     condition,
            #     n_channels_tensor)
            acts = add_and_GRU(  # GRU 这不就是wavnet的那个 GRU
                x_in,
                condition,
            )
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                if x_mask is not None:
                    x = (x + res_acts) * x_mask
                else:
                    x = x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts

        if x_mask is not None:
            out = output * x_mask
        else:
            out = output
        return out

    # def remove_weight_norm(self):
    #     if self.gin_channels != 0:
    #         remove_weight_norm_modules(self.cond_layer)
    #     for l in self.in_layers:
    #         remove_weight_norm_modules(l)
    #     for l in self.res_skip_layers:
    #         remove_weight_norm_modules(l)


class ResidualCouplingLayer(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0,
                 gin_channels=0,
                 mean_only=False,
                 wn_sharing_parameter=None  # 不明的共享权重
                 ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout,
                      gin_channels=gin_channels) if wn_sharing_parameter is None else wn_sharing_parameter
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask=None, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        if x_mask is not None:
            h = self.pre(x0) * x_mask
        else:
            h = self.pre(x0)
        h = self.enc(h, x_mask, g=g)

        if x_mask is not None:
            stats = self.post(h) * x_mask
        else:
            stats = self.post(h)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            if x_mask is not None:
                x1 = m + x1 * torch.exp(logs) * x_mask
            else:
                x1 = m + x1 * torch.exp(logs)
            # x1 = m + x1 * torch.exp(logs) * x_mask  # 逆过程
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            if x_mask is not None:
                x1 = (x1 - m) * torch.exp(-logs) * x_mask
            else:
                x1 = (x1 - m) * torch.exp(-logs)
            # x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0,
                 share_parameter=False
                 ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0,
                     gin_channels=gin_channels) if share_parameter else None

        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                      gin_channels=gin_channels, mean_only=True, wn_sharing_parameter=self.wn))
            self.flows.append(Flip())

    def forward(self, x, x_mask=None, g=None, reverse=False):
        if not reverse:
            logdet_tot = 0
            for flow in self.flows:
                x, logdet = flow(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
        else:
            logdet_tot = None
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x, logdet_tot


# class TextEncoder(nn.Module):
#     def __init__(self,
#                  out_channels,
#                  hidden_channels,
#                  kernel_size,
#                  n_layers,
#                  gin_channels=0,
#                  filter_channels=None,
#                  n_heads=None,
#                  p_dropout=None):
#         super().__init__()
#         self.out_channels = out_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.n_layers = n_layers
#         self.gin_channels = gin_channels
#         self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
#         self.f0_emb = nn.Embedding(256, hidden_channels)
#
#         self.enc_ = attentions.Encoder(
#             hidden_channels,
#             filter_channels,
#             n_heads,
#             n_layers,
#             kernel_size,
#             p_dropout)
#
#     def forward(self, x, x_mask, f0=None, noice_scale=1):
#         x = x + self.f0_emb(f0).transpose(1, 2)
#         x = self.enc_(x * x_mask, x_mask)
#         stats = self.proj(x) * x_mask
#         m, logs = torch.split(stats, self.out_channels, dim=1)
#         z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask
#
#         return z, m, logs, x_mask


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
            self,
            dim: int,
            intermediate_dim: int,
            layer_scale_init_value: Optional[float] = None, drop_path: float = 0.0, drop_out: float = 0.0

    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.dropout = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.dropout(x)

        x = residual + self.drop_path(x)
        return x


class condition_latent_encoder_att(nn.Module):
    def __init__(self, in_chans, out_channels, n_chans, n_heads, n_layers, condition_encoder_kernel_size, dropout_rate,
                 filter_channels=None):
        super().__init__()
        if filter_channels is None:
            filter_channels = n_chans * 4

        self.proj_in = nn.Conv1d(in_chans, n_chans, 1)

        self.enc = RelativeFFTBlock(hidden_channels=n_chans, filter_channels=filter_channels, n_heads=n_heads,
                                    n_layers=n_layers,
                                    kernel_size=condition_encoder_kernel_size, p_dropout=dropout_rate)

        self.proj_out = nn.Conv1d(n_chans, out_channels * 2, 1)

    def forward(self, x, noice_scale=1):
        x = self.proj_in(x)

        x = self.enc(x)
        stats = self.proj_out(x)
        m, logs = torch.chunk(stats, 2, 1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale)

        return z, m, logs,


class condition_latent_encoder_convnext(nn.Module):
    def __init__(self, in_chans, out_channels, n_chans, n_heads, n_layers, condition_encoder_kernel_size, dropout_rate,
                 filter_channels=None):
        super().__init__()
        if filter_channels is None:
            filter_channels = n_chans * 4

        self.proj_in = nn.Conv1d(in_chans, n_chans, 1)

        self.conv = nn.ModuleList(
            [ConvNeXtBlock(dim=n_chans, intermediate_dim=filter_channels, layer_scale_init_value=1e-6,
                           drop_out=dropout_rate) for _ in range(n_layers)])

        self.proj_out = nn.Conv1d(n_chans, out_channels * 2, 1)

    def forward(self, x, noice_scale=1):
        x = self.proj_in(x)

        for i in self.conv:
            x = i(x)
        stats = self.proj_out(x)
        m, logs = torch.chunk(stats, 2, 1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale)

        return z, m, logs,


class condition_encoder_att(nn.Module):
    def __init__(self, in_chans, out_channels, n_chans, n_heads, n_layers, condition_encoder_kernel_size, dropout_rate,
                 filter_channels=None):
        super().__init__()
        if filter_channels is None:
            filter_channels = n_chans * 4

        self.proj_in = nn.Conv1d(in_chans, n_chans, condition_encoder_kernel_size,
                                 padding=condition_encoder_kernel_size // 2)

        self.enc = RelativeFFTBlock(hidden_channels=n_chans, filter_channels=filter_channels, n_heads=n_heads,
                                    n_layers=n_layers,
                                    kernel_size=condition_encoder_kernel_size, p_dropout=dropout_rate)

        self.proj_out = nn.Conv1d(n_chans, out_channels, kernel_size=condition_encoder_kernel_size,
                                  padding=condition_encoder_kernel_size // 2)

    def forward(self, x, ):
        x = self.proj_in(x)

        x = self.enc(x)
        stats = self.proj_out(x)

        return stats


class condition_encoder_convnext(nn.Module):
    def __init__(self, in_chans, out_channels, n_chans, n_heads, n_layers, condition_encoder_kernel_size, dropout_rate,
                 filter_channels=None):
        super().__init__()
        if filter_channels is None:
            filter_channels = n_chans * 4

        self.proj_in = nn.Conv1d(in_chans, n_chans, condition_encoder_kernel_size,
                                 padding=condition_encoder_kernel_size // 2)

        self.conv = nn.ModuleList(
            [ConvNeXtBlock(dim=n_chans, intermediate_dim=filter_channels, layer_scale_init_value=1e-6,
                           drop_out=dropout_rate) for _ in range(n_layers)])

        self.proj_out = nn.Conv1d(n_chans, out_channels, kernel_size=condition_encoder_kernel_size,
                                  padding=condition_encoder_kernel_size // 2)

    def forward(self, x, ):
        x = self.proj_in(x)

        for i in self.conv:
            x = i(x)
        stats = self.proj_out(x)

        return stats


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self, latent_encoder_hidden_channels, latent_encoder_n_heads,
                 latent_encoder_n_layers, latent_encoder_kernel_size, latent_encoder_dropout_rate,

                 condition_in_chans,

                 condition_encoder_hidden_channels,
                 condition_encoder_n_heads,
                 condition_encoder_n_layers,
                 condition_encoder_kernel_size,
                 condition_encoder_dropout_rate,

                 inter_channels,
                 hidden_channels,

                 condition_channels,flow_wavenet_lay=4,

                 condition_encoder_filter_channels=None,

                 latent_encoder_filter_channels=None,

                 use_depthwise_conv=False,

                 flow_share_parameter=False,
                 n_flow_layer=4, latent_encoder_type='attention', use_latent_encoder=True, use_latent=True,
                 ues_condition_encoder=False, ues_condition=False, condition_encoder_type='attention',

                 **kwargs):

        super().__init__()
        self.inter_channels = inter_channels
        self.ues_condition = ues_condition

        self.use_latent = use_latent

        if use_latent_encoder and use_latent:
            if latent_encoder_type == 'attention':
                self.latent_encoder = condition_latent_encoder_att(in_chans=condition_in_chans,
                                                                   out_channels=inter_channels,
                                                                   n_chans=latent_encoder_hidden_channels,
                                                                   n_heads=latent_encoder_n_heads,
                                                                   n_layers=latent_encoder_n_layers,
                                                                   condition_encoder_kernel_size=latent_encoder_kernel_size,
                                                                   dropout_rate=latent_encoder_dropout_rate,
                                                                   filter_channels=latent_encoder_filter_channels)
            elif latent_encoder_type == 'convnext':
                self.latent_encoder = condition_latent_encoder_convnext(in_chans=condition_in_chans,
                                                                        out_channels=inter_channels,
                                                                        n_chans=latent_encoder_hidden_channels,
                                                                        n_heads=None,
                                                                        n_layers=latent_encoder_n_layers,
                                                                        condition_encoder_kernel_size=None,
                                                                        dropout_rate=latent_encoder_dropout_rate,
                                                                        filter_channels=latent_encoder_filter_channels)
            else:
                raise RuntimeError("unsupport_latent_encoder")

        elif ((not use_latent_encoder) and use_latent):
            self.condition_encoder = nn.Conv1d(condition_in_chans, inter_channels, kernel_size=7, padding=3)

        if ues_condition_encoder and ues_condition:
            if condition_encoder_type == 'attention':
                self.condition_encoder = condition_encoder_att(in_chans=condition_in_chans,
                                                               out_channels=condition_channels,
                                                               n_chans=condition_encoder_hidden_channels,
                                                               n_heads=condition_encoder_n_heads,
                                                               n_layers=condition_encoder_n_layers,
                                                               condition_encoder_kernel_size=condition_encoder_kernel_size,
                                                               dropout_rate=condition_encoder_dropout_rate,
                                                               filter_channels=condition_encoder_filter_channels)
            elif condition_encoder_type == 'convnext':
                self.condition_encoder = condition_encoder_convnext(in_chans=condition_in_chans,
                                                                    out_channels=condition_channels,
                                                                    n_chans=condition_encoder_hidden_channels,
                                                                    n_heads=None,
                                                                    n_layers=condition_encoder_n_layers,
                                                                    condition_encoder_kernel_size=condition_encoder_kernel_size,
                                                                    dropout_rate=condition_encoder_dropout_rate,
                                                                    filter_channels=condition_encoder_filter_channels)
            else:
                raise RuntimeError("unsupport__encoder")
        elif ((not ues_condition_encoder) and ues_condition):
            self.condition_encoder = nn.Conv1d(condition_in_chans, condition_channels, kernel_size=7, padding=3)

        self.use_depthwise_conv = use_depthwise_conv

        # self.enc_p = TextEncoder(
        #     inter_channels,
        #     hidden_channels,
        #     filter_channels=filter_channels,
        #     n_heads=n_heads,
        #     n_layers=n_layers,
        #     kernel_size=kernel_size,
        #     p_dropout=p_dropout
        # )

        set_Conv1dModel(self.use_depthwise_conv)

        if ues_condition:
            condition_channelsw = condition_channels
        else:
            condition_channelsw = 0

        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flow_layer,n_flows=flow_wavenet_lay,
                                          gin_channels=condition_channelsw, share_parameter=flow_share_parameter)

    def forward(self, c, mel, x_mask=None):

        # vol proj

        # f0 predict

        # encoder
        if self.use_latent:
            z_ptemp, m_p, logs_p = self.latent_encoder(c)
        else:
            m_p, logs_p = None, None
        # z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))

        # flow
        if self.ues_condition:
            condition = self.condition_encoder(c)
            z_p, logdet = self.flow(mel, x_mask, g=condition)
        else:
            z_p, logdet = self.flow(mel, x_mask, g=None)

        return x_mask, (z_p, m_p, logs_p), logdet,

    @torch.no_grad()
    def infer(self, c, noice_scale=0.35, seed=None, ):
        if seed is not None:

            if c.device == torch.device("cuda"):
                torch.cuda.manual_seed_all(seed)
            else:
                torch.manual_seed(seed)

        if self.use_latent:
            z_p, m_p, logs_p = self.latent_encoder(c)
        else:
            z_p = torch.randn_like(torch.zeros(1, self.inter_channels, c.size()[2])) * noice_scale

            z_p=z_p.cuda()

        # vol proj

        # z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale)
        # o, _ = self.flow(z_p,  g=g, reverse=True)

        if self.ues_condition:
            condition = self.condition_encoder(c)
            # z_p, logdet = self.flow(mel, x_mask, g=condition)
            o, _ = self.flow(z_p, g=condition, reverse=True)
        else:
            o, _ = self.flow(z_p, g=None, reverse=True)

        return o


class glow_loss_L(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pack_loss,target):

        z, m, logs, logdet, mask = pack_loss
        # z, m, logs, logdet, mask = None

        l = 0.5 * torch.sum(
            torch.exp(-2 * logdet) * ((z ) ** 2))  # neg normal likelihood w/o the constant term
        l = l - torch.sum(logdet)  # log jacobian determinant
        if mask is not None:
            l = l / torch.sum(torch.ones_like(z) * mask)  # averaging across batch, channel and time axes
        else:
            l = l / torch.sum(torch.ones_like(z))  # averaging across batch, channel and time axes
        l = l + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
        return l





class glow_decoder(nn.Module):
    def __init__(self, encoder_hidden, out_dims, latent_encoder_hidden_channels, latent_encoder_n_heads,
                 latent_encoder_n_layers, latent_encoder_kernel_size, latent_encoder_dropout_rate,
                 condition_encoder_hidden_channels, condition_encoder_n_heads, condition_encoder_n_layers,
                 condition_encoder_kernel_size, condition_encoder_dropout_rate, flow_hidden_channels,
                 flow_condition_channels, parame,use_mask=True,use_norm=True,flow_wavenet_lay=4,flow_infer_seed=None,flow_infer_scale=0.35,
                 condition_encoder_filter_channels=None,

                 latent_encoder_filter_channels=None,

                 use_depthwise_conv=False,

                 flow_share_parameter=False,
                 n_flow_layer=4, latent_encoder_type='attention', use_latent_encoder=True,
                 use_latent=True,
                 ues_condition_encoder=False, ues_condition=False,
                 condition_encoder_type='attention'):
        super().__init__()
        self.use_latent=use_latent
        self.flow_infer_seed=flow_infer_seed
        self.flow_infer_scale=flow_infer_scale
        self.glow_decoder = SynthesizerTrn(latent_encoder_hidden_channels=latent_encoder_hidden_channels,
                                           latent_encoder_n_heads=latent_encoder_n_heads,
                                           latent_encoder_n_layers=latent_encoder_n_layers,
                                           latent_encoder_kernel_size=latent_encoder_kernel_size,
                                           latent_encoder_dropout_rate=latent_encoder_dropout_rate,

                                           condition_in_chans=encoder_hidden,

                                           condition_encoder_hidden_channels=condition_encoder_hidden_channels,
                                           condition_encoder_n_heads=condition_encoder_n_heads,
                                           condition_encoder_n_layers=condition_encoder_n_layers,
                                           condition_encoder_kernel_size=condition_encoder_kernel_size,
                                           condition_encoder_dropout_rate=condition_encoder_dropout_rate,

                                           inter_channels=out_dims,
                                           flow_wavenet_lay=flow_wavenet_lay,
                                           hidden_channels=flow_hidden_channels,

                                           condition_channels=flow_condition_channels,

                                           condition_encoder_filter_channels=condition_encoder_filter_channels,

                                           latent_encoder_filter_channels=latent_encoder_filter_channels,

                                           use_depthwise_conv=use_depthwise_conv,

                                           flow_share_parameter=flow_share_parameter,
                                           n_flow_layer=n_flow_layer, latent_encoder_type=latent_encoder_type,
                                           use_latent_encoder=use_latent_encoder,
                                           use_latent=use_latent,
                                           ues_condition_encoder=ues_condition_encoder, ues_condition=ues_condition,
                                           condition_encoder_type=condition_encoder_type)

        self.use_mask=use_mask
        self.use_norm=use_norm

    def norm(self,x):
        x = (x - (-5)) / (0 - (-5)) * 2 - 1
        return x

    def denorm(self,x):
        x=(x + 1) / 2 * (0 - (-5)) + (-5)
        return x

    def build_loss(self):


        if self.use_latent:

            return glow_loss_L()

        return glow_loss_L()
    def forward(self, x, infer, x_gt,mask):
        if not self.use_mask or infer:
            mask=None
        else:
            mask=mask.transpose(1, 2)




        if infer:
            out=self.glow_decoder.infer(x.transpose(1, 2), noice_scale=self.flow_infer_scale, seed=self.flow_infer_seed).transpose(1, 2)
            if self.use_norm:
                out = self.denorm(out)
            return out
        else:
            if self.use_norm:
                x_gt = self.norm(x_gt)


            x = x.transpose(1, 2)
            x_gt=x_gt.transpose(1, 2)

            x_mask, (z_p, m_p, logs_p), logdet=self.glow_decoder(x,x_gt,x_mask=mask)


            pack_loss = (z_p, m_p, logs_p, logdet, x_mask)
            return pack_loss




        pass
