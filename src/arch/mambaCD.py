try:
    from .MambaCD.changedetection.models.STMambaSCD import STMambaSCD
    from .MambaCD.changedetection.configs.config import get_config
except:
    print("No MambaCD : Import error !")

import torch
import os

def makeModel(args):
    ckpt_url = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth"
    if not os.path.isfile("checkpoints/vssm_base_0229_ckpt_epoch_237.pth"):
        print(f"Downloading VMamba base checkpoint from {ckpt_url}. Saving into checkpoints/")
        torch.hub.download_url_to_file("https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth","checkpoints/")

    config = get_config(args)
    model = STMambaSCD(output_cd = 2, 
                output_clf = args.classes,
                pretrained="checkpoints/vssm_base_0229_ckpt_epoch_237.pth",
                patch_size=config.MODEL.VSSM.PATCH_SIZE, 
                in_chans=args.in_channels, 
                num_classes=args.classes, 
                depths=config.MODEL.VSSM.DEPTHS, 
                dims=config.MODEL.VSSM.EMBED_DIM, 
                ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
                ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
                ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
                ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
                ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
                ssm_conv=config.MODEL.VSSM.SSM_CONV,
                ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
                ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
                ssm_init=config.MODEL.VSSM.SSM_INIT,
                forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
                mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
                mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
                mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                patch_norm=config.MODEL.VSSM.PATCH_NORM,
                norm_layer=config.MODEL.VSSM.NORM_LAYER,
                downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
                patchembed_version=config.MODEL.VSSM.PATCHEMBED,
                gmlp=config.MODEL.VSSM.GMLP,
                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )
    return model