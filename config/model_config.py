from dotwiz import DotWiz

config_transcalib_LVT_efficientnet_june17  = {
        'model_name': 'TransCalib_LVT_EfficientNet_june17',
        'feature_matching' : {
                 'in_ch' : 512,  
                 'conv_repeat' : 2,
                 'conv_act' : 'elu',
                 'conv_drop' : 0.05,
                 'depthwise':[True, True, True],
                 'attn_repeat': [2, 2, 2],
                 'attn_types': ['csa', 'csa', 'csa'],
                 'attn_depths' : [1, 1, 1],
                 'embed_ch' : [512, 512, 2048],
                 'num_heads' : [4, 4, 8],
                 'mlp_ratios' : [4, 4, 4],  
                 'mlp_depconv' : [True, True, True], 
                 'attn_drop_rate' : 0.05, 
                 'drop_path_rate' : 0.05, 
                },
        'regression_drop': 0.1
    }