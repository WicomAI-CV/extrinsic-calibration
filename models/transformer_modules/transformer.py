# from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile
from .patch import extract_image_patches, reduce_mean, reduce_sum, same_padding, reverse_patches
import pdb
import math


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# MLP in the paper
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Efficient_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, split_segment=4, attn_drop=0., proj_drop=0.):
        super(Efficient_Self_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.s = split_segment

        self.reduce = nn.Linear(dim, dim // 2,
                                bias=qkv_bias)  # nn.Linear鍏ㄨ繛鎺ュ眰鍏惰緭鍏ヨ緭鍑轰负浜岀淮寮犻噺锛岄渶瑕佸皢4缁村紶閲忚浆鎹负浜岀淮寮犻噺涔嬪悗鎵嶈兘浣滀负杈撳叆
        self.W_qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.W_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # feature split of the Efficient Transformer
        q_all = torch.split(q, math.ceil(N // self.s), dim=-2)
        k_all = torch.split(k, math.ceil(N // self.s), dim=-2)
        v_all = torch.split(v, math.ceil(N // self.s), dim=-2)

        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale  # 16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2)  # .reshape(B, N, C)
            output.append(trans_x)
        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Efficient_Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, split_segment=4, attn_drop=0., proj_drop=0.):
        super(Efficient_Cross_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.s = split_segment

        self.reduce = nn.Linear(dim, dim // 2,
                                bias=qkv_bias)  # nn.Linear鍏ㄨ繛鎺ュ眰鍏惰緭鍏ヨ緭鍑轰负浜岀淮寮犻噺锛岄渶瑕佸皢4缁村紶閲忚浆鎹负浜岀淮寮犻噺涔嬪悗鎵嶈兘浣滀负杈撳叆
        
        self.W_q = nn.Linear(dim // 2, dim // 2, bias=qkv_bias)
        self.W_k = nn.Linear(dim // 2, dim // 2, bias=qkv_bias)
        self.W_v = nn.Linear(dim // 2, dim // 2, bias=qkv_bias)
        
        self.proj = nn.Linear(dim // 2, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_query, x_kv):
        x_query = self.reduce(x_query)
        x_kv = self.reduce(x_kv)
        B, N, C = x_kv.shape
        B_, N_, C_ = x_query.shape
        kv = self.W_kv(x_kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.W_q(x_query).reshape(B_, N_, 1, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)

        # feature split of the Efficient Transformer
        q_all = torch.split(q, math.ceil(N // self.s), dim=-2)
        k_all = torch.split(k, math.ceil(N // self.s), dim=-2)
        v_all = torch.split(v, math.ceil(N // self.s), dim=-2)

        print("patch size", q_all[0].shape, v_all[0].shape)

        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale  # 16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2)  # .reshape(B, N, C)
            output.append(trans_x)
        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



# Efficient Multi-Head Attention in the paper
class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim//9, bias=qkv_bias) 
        self.qkv = nn.Linear(dim//9, dim * 3//9, bias=qkv_bias)
        self.proj = nn.Linear(dim//9, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



## Key Module: Efficient Transformer (ET) in the paper
# class TransBlock(nn.Module):
#     def __init__(
#             self, n_feat=32, dim=288, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#             drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
#         super(TransBlock, self).__init__()
#         self.dim = dim
#         self.atten = EffAttention(self.dim, num_heads=8, qkv_bias=False, qk_scale=None,
#                                   attn_drop=0., proj_drop=0.)
#         self.norm1 = nn.LayerNorm(self.dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=dim // 4, act_layer=act_layer, drop=drop)
#         self.norm2 = nn.LayerNorm(self.dim)

#     def forward(self, x):
#         B = x.shape[0]
#         x = extract_image_patches(x, ksizes=[3, 3],
#                                   strides=[1, 1],
#                                   rates=[1, 1],
#                                   padding='same')  # 16*2304*576
#         x = x.permute(0, 2, 1)

#         x = x + self.atten(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

class Transformer_MHSA(nn.Module):
    def __init__(
            self, n_feat=32, ksize=[3, 3], num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(Transformer_MHSA, self).__init__()
        self.dim = n_feat*ksize[0]*ksize[1]
        self.ksize = ksize
        self.atten = Efficient_Self_Attention(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                attn_drop=attn_drop, proj_drop=0.)
        self.norm1 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(in_features=self.dim, hidden_features=self.dim // mlp_ratio, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x):
        B, C, H, W = x.shape
        print(x.shape)
        x = extract_image_patches(x, ksizes=self.ksize,
                                  strides=[1, 1],
                                  rates=[1, 1],
                                  padding='same')  # 16*2304*576
        print(x.shape)
        x = x.permute(0, 2, 1)

        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 2, 1)
        x = reverse_patches(x, (H, W), (3, 3), 1, 1)

        return x
    
class Transformer_MHCA(nn.Module):
    def __init__(
            self, n_feat=32, ksize=[7, 7], num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(Transformer_MHCA, self).__init__()
        self.dim = n_feat*ksize[0]*ksize[1]
        self.ksize = ksize
        self.atten = Efficient_Cross_Attention(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=0.)
        self.norm1 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(in_features=self.dim, hidden_features=self.dim // mlp_ratio, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x1, x2):
        B, C, H, W = x2.shape
        print("x1 large",x1.shape)
        x_q = extract_image_patches(x1, ksizes=self.ksize,
                                  strides=[1, 1],
                                  rates=[1, 1],
                                  padding='same')  # 16*2304*576
        print("x1 patched",x_q.shape)
        x_kv = extract_image_patches(x2, ksizes=self.ksize,
                                  strides=[1, 1],
                                  rates=[1, 1],
                                  padding='same') 
        print("x2 patched",x_kv.shape)
        x_q = x_q.permute(0, 2, 1)
        x_kv = x_kv.permute(0, 2, 1)

        x = x_kv + self.atten(self.norm1(x_q), self.norm1(x_kv))
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 2, 1)
        x = reverse_patches(x, (H, W), (3, 3), 1, 1)
        
        return x
