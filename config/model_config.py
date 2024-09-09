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

config_lvt_effnet_light_v1_july18  = {
        'model_name': 'TransCalib_LVT_EfficientNet_july18',
        'feature_matching' : {
                 'in_ch' : 512,  
                 'conv_repeat' : 1,
                 'conv_act' : 'elu',
                 'conv_drop' : 0.05,
                 'depthwise':[True, True, True],
                 'attn_repeat': [1, 1, 1],
                 'attn_types': ['csa', 'csa', 'csa'],
                 'attn_depths' : [1, 1, 1],
                 'embed_ch' : [512, 512, 512],
                 'num_heads' : [4, 8, 16],
                 'mlp_ratios' : [4, 4, 4],  
                 'mlp_depconv' : [True, True, True], 
                 'attn_drop_rate' : 0.05, 
                 'drop_path_rate' : 0.05, 
                },
        'regression_drop': 0.1
    }

config_lvt_LETnet_light_v1_july19  = {
        'model_name': 'TransCalib_LVT_LETNet_july19',
        'feature_matching' : {
                 'in_ch' : 64,  
                 'conv_repeat' : 2,
                 'conv_act' : 'elu',
                 'conv_drop' : 0.05,
                 'depthwise':[True, True, True],
                 'attn_repeat': [2, 2, 2],
                 'attn_types': ['csa', 'csa', 'csa'],
                 'attn_depths' : [1, 1, 1],
                 'embed_ch' : [128, 256, 512],
                 'num_heads' : [4, 8, 16],
                 'mlp_ratios' : [4, 4, 4],  
                 'mlp_depconv' : [True, True, True], 
                 'attn_drop_rate' : 0.05, 
                 'drop_path_rate' : 0.05, 
                },
        'regression_drop': 0.1
    }

config_transcalib_LVT_efficientnet_ablation  = {
        'model_name': 'TransCalib_LVT_EfficientNet_ablation',
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