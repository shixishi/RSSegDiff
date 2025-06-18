# 通过build_model函数创建模型对象
from .nn_unet import NNUNet
from .cn_unet import ConditionUNet
from .cnfsegmodel import CnfSegModel, FeatureFusionModule

def build_model(config, train_flag):
    # 1 构建条件网络
    cond_cfg = config.get("cond_net", {})
    cond_net = ConditionUNet(
    in_channels = cond_cfg["in_channels"],
    model_channels = cond_cfg["model_channels"],
    out_channels = cond_cfg["out_channels"],
    num_res_blocks = cond_cfg["num_res_blocks"],
    attention_resolutions = cond_cfg["attention_resolutions"],
    dropout=cond_cfg["dropout"],
    channel_mult = cond_cfg["channel_mult"]
    )

    # 2 构建去噪网络
    noise_cfg = config.get("noise_net", {})
    noise_net = NNUNet(
        in_channels = noise_cfg["in_channels"],
        model_channels = noise_cfg["model_channels"],
        out_channels = noise_cfg["out_channels"],
        num_res_blocks = noise_cfg["num_res_blocks"],
        attention_resolutions = noise_cfg["attention_resolutions"],
        dropout= noise_cfg["dropout"],
        channel_mult = noise_cfg["channel_mult"]
    )
    # 3) 构建融合模块
    ffm_cfg = config.get("ffm", {})
    ffm = FeatureFusionModule(
        in_c_cn = ffm_cfg["in_c_cn"],
        in_c_nn = ffm_cfg["in_c_nn"],
        proj_c = ffm_cfg["proj_c"],
        num_heads = ffm_cfg["num_heads"],
        ffn_ratio = ffm_cfg["ffn_mult"],
    )

    # 4) 包装成 CnfSegModel
    model = CnfSegModel(cond_net=cond_net, noise_net=noise_net, ffm=ffm, train_flag=train_flag)

    return model