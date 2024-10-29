import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from ..realignment_layer import realignment_layer

from torchvision.models import (efficientnet_v2_s)

from .lvt import (Transformer_block, OverlapPatchEmbed)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def conv3x3(in_ch, out_ch, strd, pad, grp):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=strd, padding=pad, groups=grp)

def conv1x1(in_ch, out_ch, strd, pad, grp):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=strd, padding=pad, groups=grp)

class conv3x3_BN_act(nn.Module):
    def __init__(self, in_ch, out_ch, 
                 stride, padding=0, group=1, act='elu', drop_rate = 0.1 ,**kwargs):
        super(conv3x3_BN_act, self).__init__()
        self.conv = conv3x3(in_ch, out_ch, stride, padding, group)
        self.BN = nn.BatchNorm2d(out_ch)
        self.slope = kwargs.get("slope", 0.01)
        if act == 'elu':
            self.act = nn.ELU(inplace=True)
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU(self.slope, inplace=True)
        else:
            self.act = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.BN(x)
        x = self.dropout(x)

        return x

class conv1x1_BN_act(nn.Module):
    def __init__(self, in_ch, out_ch, 
                 stride, padding=0, group=1, act='elu', drop_rate = 0.1 ,**kwargs):
        super(conv1x1_BN_act, self).__init__()
        self.conv = conv1x1(in_ch, out_ch, stride, padding, group)
        self.BN = nn.BatchNorm2d(out_ch)
        self.slope = kwargs.get("slope", 0.01)
        if act == 'elu':
            self.act = nn.ELU(inplace=True)
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU(self.slope, inplace=True)
        else:
            self.act = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.BN(x)
        x = self.dropout(x)

        return x

class conv_block_new(nn.Module):
    def __init__(self, in_ch, out_ch, depthwise=True, act='elu', drop_rate = 0.1):
        super(conv_block_new, self).__init__()
        self.depthwise = depthwise
        
        self.conv1 = conv3x3_BN_act(in_ch, in_ch, 
                                          1, padding=1, group=1, act=act, drop_rate=drop_rate)
        if depthwise:
            assert out_ch % in_ch == 0
            self.conv2 = conv3x3_BN_act(in_ch, in_ch, 
                                          1, padding=1, group=in_ch, act=act, drop_rate=drop_rate)
            
            self.conv3 = conv1x1_BN_act(in_ch, out_ch, 1, padding=0, group=out_ch, act=act, drop_rate=drop_rate)
        else:
            self.conv2 = conv3x3_BN_act(in_ch, in_ch, 
                                          1, padding=1, group=1, act=act, drop_rate=drop_rate)
            self.conv3 = conv1x1_BN_act(in_ch, out_ch, 1, padding=0, group=1, act=act, drop_rate=drop_rate)

    def forward(self, x):
        skip1 = x
        x = self.conv1(x)
        skip2 = x
        x = self.conv2(x)
        x = x + skip1 + skip2
        x = self.conv3(x)
        return x
    
class conv_attn_csa_new(nn.Module):
    def __init__(self, 
                 in_ch = 3,  
                 # conv config
                 conv_repeat = 3,
                 conv_act = 'elu',
                 conv_drop = 0.1,
                 depthwise=[False, True, True],
                 # transformer config
                 attn_repeat = [1, 1, 1],
                 attn_types = ['csa', 'csa', 'csa'],
                 attn_depths = [2, 2, 2],
                 embed_ch =[64, 128, 256],
                 num_heads = [2, 2, 4],
                 mlp_ratios=[4, 4, 4], 
                 mlp_depconv=[True, True, True], 
                 sr_ratios=[1,1,1], 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 ):
        super(conv_attn_csa_new, self).__init__()

        self.depthwise = depthwise
        self.attn_depths = attn_depths
        
        self.repeat = len(embed_ch)
        self.repeated_conv_blocks = nn.ModuleList()
        self.repeated_pe_attns = nn.ModuleList()
        
        for i in range(self.repeat):
            conv_blocks = nn.ModuleList()
            for _ in range(conv_repeat):
                in_channel = in_ch if i == 0 else embed_ch[i-1]
                conv_blocks.append(conv_block_new(in_channel, in_channel, depthwise[i], conv_act, conv_drop))

            pe_attns = nn.ModuleList()
            for k in range(attn_repeat[i]):
                if k == 0:
                    stride = 2
                    in_dim = in_ch if i == 0 else embed_ch[i-1]
                else:
                    stride = 1
                    in_dim = embed_ch[i]
                _patch_embed = OverlapPatchEmbed(
                    patch_size=3,
                    stride=stride,
                    in_chans=in_dim,
                    embed_dim=embed_ch[i],
                )

                _blocks = []
                for l in range(attn_depths[i]):
                    block_dpr = drop_path_rate * (l + sum(attn_depths[:i])) / (sum(attn_depths) - 1)
                    _blocks.append(Transformer_block(
                        embed_ch[i], 
                        num_heads=num_heads[i], 
                        mlp_ratio=mlp_ratios[i],
                        sa_layer=attn_types[i],
                        rasa_cfg=None, # I am here
                        sr_ratio=sr_ratios[i],
                        qkv_bias=qkv_bias, qk_scale=qk_scale, 
                        attn_drop=attn_drop_rate, drop_path=block_dpr,
                        with_depconv=mlp_depconv[i]))
                _blocks = nn.Sequential(*_blocks)
                
                pe_attns.append(nn.Sequential(
                    _patch_embed, 
                    _blocks
                ))
            
            self.repeated_conv_blocks.append(conv_blocks)
            self.repeated_pe_attns.append(pe_attns)
        
        self.downstream_norms = nn.ModuleList([norm_layer(embed_dim) 
                                                   for embed_dim in embed_ch])
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for i in range(self.repeat):
            for conv in self.repeated_conv_blocks[i]:
                x = conv(x)
                # print("conv out:", x.shape)
            x = x.permute(0,2,3,1)
            for pe_attns in self.repeated_pe_attns[i]:
                for layer in pe_attns:
                    x = layer(x)
                    x = self.downstream_norms[i](x)
                    # print("attn out:", x.shape)
            x = x.permute(0,3,1,2)
            # print("conv attn out:", x.shape)

        return(x)

class TransCalib_lvt_efficientnet_june2(nn.Module):
    def __init__(self, model_config, trans_norm=False):
        super(TransCalib_lvt_efficientnet_june2, self).__init__()
    
        self.trans_norm = trans_norm

        self.featmat_config = model_config.feature_matching
        self.regression_dropout = model_config.regression_drop
        
        self.rgb_branch = nn.Sequential(*list(
                    efficientnet_v2_s(weights='IMAGENET1K_V1').features.children())[:-1])
        
        self.depth_branch = nn.Sequential(*list(
                    efficientnet_v2_s(weights='IMAGENET1K_V1').features.children())[:-1])
        self.depth_branch[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.feature_matching = conv_attn_csa_new(
                 in_ch = self.featmat_config.in_ch,  
                 conv_repeat = self.featmat_config.conv_repeat,
                 conv_act = self.featmat_config.conv_act,
                 conv_drop = self.featmat_config.conv_drop,
                 depthwise= self.featmat_config.depthwise,
                 attn_repeat = self.featmat_config.attn_repeat,
                 attn_types = self.featmat_config.attn_types,
                 attn_depths =  self.featmat_config.attn_depths,
                 embed_ch = self.featmat_config.embed_ch,
                 num_heads =  self.featmat_config.num_heads,
                 mlp_ratios = self.featmat_config.mlp_ratios, 
                 mlp_depconv = self.featmat_config.mlp_depconv, 
                 attn_drop_rate = self.featmat_config.attn_drop_rate, 
                 drop_path_rate = self.featmat_config.drop_path_rate,   
                 )
        
        self.regression_head = default_regression_head(self.regression_dropout)
        self.recalib = realignment_layer()

    def forward(self, rgb_im, depth_im, pcd_mis, T_mis_batch):
        x_rgb = self.rgb_branch(rgb_im)
        x_depth = self.depth_branch(depth_im)

        # print("rgb shape:", x_rgb.shape)
        # print("depth shape:", x_depth.shape)

        x_out = torch.cat((x_rgb, x_depth), dim=1)

        # print(x_out.shape)
        x_out = self.feature_matching(x_out)
        # print("x_out after featmat: ", x_out.shape)

        delta_t_pred, delta_q_pred = self.regression_head(x_out)

        if delta_q_pred.ndim < 2:
            delta_q_pred = torch.unsqueeze(delta_q_pred, 0)
            delta_t_pred = torch.unsqueeze(delta_t_pred, 0)

        delta_q_pred = nn.functional.normalize(delta_q_pred)

        # # print(delta_t_pred.shape, delta_q_pred.shape)

        batch_T_pred, pcd_pred = self.recalib(pcd_mis, T_mis_batch, delta_q_pred, delta_t_pred)

        return pcd_pred, batch_T_pred, delta_q_pred, delta_t_pred
    
class default_regression_head(nn.Module):
    def __init__(self, dropout=0.0):
        super(default_regression_head, self).__init__()
        self.dropout = dropout
        self.fc = nn.Sequential(nn.Linear(2048*3,1024),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(1024,512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512,256),
                                nn.ReLU())
        self.fc_trans = nn.Sequential(nn.Linear(256,128),
                                      nn.ReLU(),
                                      nn.Linear(128,64),
                                      nn.ReLU(),
                                      nn.Linear(64,3))
        self.fc_rot = nn.Sequential(nn.Linear(256,128),
                                    nn.ReLU(),
                                    nn.Linear(128,64),
                                    nn.ReLU(),
                                    nn.Linear(64,4))
    
    def forward(self, x):
        # print("head input size", x.shape)
        x = torch.flatten(x, 1)
        # print("head squeezed size", x.shape)
        x = self.fc(x)
        # print("head fc size", x.shape)
        x_trans = self.fc_trans(x)
        x_rot = self.fc_rot(x)

        return x_trans, x_rot