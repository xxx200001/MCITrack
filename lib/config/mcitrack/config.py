from easydict import EasyDict as edict
import yaml

'''
Gohan: Dinov2 combined with One-stream framework.
'''

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "/pretrained/MCITRACK_ep0300.pth.tar"
cfg.MODEL.HIDDEN_DIM = 512  #新加512维度


# LANGUAGE
cfg.MODEL.LANGUAGE = edict()
cfg.MODEL.LANGUAGE.IMPLEMENT = 'pytorch'
cfg.MODEL.LANGUAGE.TYPE = 'bert-base-uncased'
cfg.MODEL.LANGUAGE.PATH = 'pretrained/bert/bert-base-uncased.tar.gz'
cfg.MODEL.LANGUAGE.VOCAB_PATH = 'pretrained/bert/bert-base-uncased-vocab.txt'
cfg.MODEL.LANGUAGE.BERT = edict()
cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN = 30


# MODEL.ENCODER
# for more customization for encoder, please modify lib/models/mcitrack/vit.py
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "dinov2_vitb14" # encoder model
cfg.MODEL.ENCODER.DROP_PATH = 0
cfg.MODEL.ENCODER.PRETRAIN_TYPE = "mae" #  mae, default, or scratch. This parameter is not activated for dinov2.
cfg.MODEL.ENCODER.PRETRAIN_TYPE1 = "mae"
cfg.MODEL.ENCODER.USE_CHECKPOINT = False # to save the memory.
cfg.MODEL.ENCODER.STRIDE = 14
cfg.MODEL.ENCODER.POS_TYPE = 'interpolate' # type of loading the positional encoding. "interpolate" or "index".
cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE = False # add a token_type_embedding to indicate the search, template_foreground, template_background
cfg.MODEL.ENCODER.INTERACTION_INDEXES = [[0, 6], [6, 12], [12, 18], [18, 24]]
cfg.MODEL.ENCODER.GRAD_CKPT = False
# MODEL.NECK
cfg.MODEL.NECK = edict()
cfg.MODEL.NECK.N_LAYERS = 4
cfg.MODEL.NECK.D_MODEL = 512
cfg.MODEL.NECK.D_STATE = 16 #MAMABA_HIDDEN_STATE
# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.TYPE = "CENTER" # MLP, CORNER, CENTER
cfg.MODEL.DECODER.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1  # encoder's LR = this factor * LR
cfg.TRAIN.FREEZE_ENCODER = False # for freezing the parameters of encoder
cfg.TRAIN.ENCODER_OPEN = [] # only for debug, open some layers of encoder when FREEZE_ENCODER is True
cfg.TRAIN.CE_WEIGHT = 1.0 # weight for cross-entropy loss
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.PRINT_INTERVAL = 50 # interval to print the training log
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.FIX_BN = False
cfg.TRAIN.ENCODER_W = ""
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.TYPE = "normal" # normal, peft or fft
cfg.TRAIN.PRETRAINED_PATH = None

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = "order"
cfg.DATA.LOADER = "tracking"
# cfg.DATA.MULTI_MODAL_VISION = True # vision multi-modal
cfg.DATA.MULTI_MODAL_LANGUAGE = True # language multi-modalF
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1  #number of search region, only support 1 for now.
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATEF
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 4.0
cfg.TEST.TEMPLATE_SIZE = 256
cfg.TEST.SEARCH_FACTOR = 2.0
cfg.TEST.SEARCH_SIZE = 128
cfg.TEST.EPOCH = 500
cfg.TEST.WINDOW = False # window penalty
cfg.TEST.NUM_TEMPLATES = 1

cfg.TEST.UPT = edict()
cfg.TEST.UPT.DEFAULT = 1
cfg.TEST.UPT.LASOT = 0
cfg.TEST.UPT.LASOT_EXTENSION_SUBSET = 0
cfg.TEST.UPT.TRACKINGNET = 0
cfg.TEST.UPT.TNL2K = 0
cfg.TEST.UPT.NFS = 0
cfg.TEST.UPT.UAV = 0
cfg.TEST.UPT.VOT20 = 0
cfg.TEST.UPT.GOT10K_TEST = 0

cfg.TEST.UPH = edict()
cfg.TEST.UPH.DEFAULT = 1
cfg.TEST.UPH.LASOT = 0
cfg.TEST.UPH.LASOT_EXTENSION_SUBSET = 0
cfg.TEST.UPH.TRACKINGNET = 0
cfg.TEST.UPH.TNL2K = 0
cfg.TEST.UPH.NFS = 0
cfg.TEST.UPH.UAV = 0
cfg.TEST.UPH.VOT20 = 0
cfg.TEST.UPH.GOT10K_TEST = 0

cfg.TEST.INTER = edict()
cfg.TEST.INTER.DEFAULT = 999999
cfg.TEST.INTER.LASOT = 0
cfg.TEST.INTER.LASOT_EXTENSION_SUBSET = 0
cfg.TEST.INTER.TRACKINGNET = 0
cfg.TEST.INTER.TNL2K = 0
cfg.TEST.INTER.NFS = 0
cfg.TEST.INTER.UAV = 0
cfg.TEST.INTER.VOT20 = 0
cfg.TEST.INTER.GOT10K_TEST = 0

cfg.TEST.MB = edict()
cfg.TEST.MB.DEFAULT = 500
cfg.TEST.MB.LASOT = 0
cfg.TEST.MB.LASOT_EXTENSION_SUBSET = 0
cfg.TEST.MB.TRACKINGNET = 0
cfg.TEST.MB.TNL2K = 0
cfg.TEST.MB.NFS = 0
cfg.TEST.MB.UAV = 0
cfg.TEST.MB.VOT20 = 0
cfg.TEST.MB.GOT10K_TEST = 0

# === 方向 C: 一致性模块配置 ===
# P0 (默认): consistency 仅作为诊断信号输出, 不影响任何决策
# P1: 当 USE_CONSISTENCY_FOR_H_RESET=True 时, 用 combined_score 触发 h_state reset
cfg.TEST.USE_CONSISTENCY_FOR_H_RESET = False
cfg.TEST.CONSISTENCY_RESET_ALPHA = 0.5  # conf_score 权重, (1-alpha) 为 (标定后) consistency 权重
cfg.TEST.CKPT_NAME = ''  # ablation yaml 用: 指定 checkpoint 目录名 (空字符串 = 使用 yaml_name)
# [修复 ②] consistency 与 conf_score 量纲不同 (且偏低)，直接线性相加会系统性拉低 combined_score。
# 用序列内 EMA 均值/方差做 z-score + sigmoid，把 consistency 标定到与 conf 可比的 (0,1) 尺度。
cfg.TEST.CONSISTENCY_CALIB_MOMENTUM = 0.95  # 在线标定 EMA 动量 (越大越平滑)

# === 诊断日志 (默认关; 开启后每个序列写一个 CSV: 逐帧 conf / consistency / 模板替换等) ===
cfg.TEST.LOG_DIAGNOSTICS = False
cfg.TEST.DIAG_DIR = './diag_logs'

# === TemplateSearchConsistency (无参循环一致性) 配置 ===
cfg.MODEL.CONSISTENCY = edict()
# [修复 ①] 只聚合一致性最高的 Top-K 比例 token (聚焦目标区, 防背景稀释); <=0 或 >=1 退化为全均值
cfg.MODEL.CONSISTENCY.TOPK_RATIO = 0.25
cfg.MODEL.CONSISTENCY.PARA_FACTOR = 10.0  # tanh 放大因子

# === TemplateAwareFusion 模块配置 ===
# 从 ParaHydra+ PMIFM 迁移的多模板融合模块
# ENABLE=False 时代码逻辑与 baseline 完全一致
cfg.MODEL.FUSION = edict()
cfg.MODEL.FUSION.ENABLE = False           # 是否启用融合模块
cfg.MODEL.FUSION.SCALE_FACTOR = 24.0      # 对应 PMIFM 的 scale_factor (softmax 前乘, 控制模板权重对比度)
cfg.MODEL.FUSION.PARA_FACTOR = 10.0       # 对应 tanh 放大因子 (二值化 V)
cfg.MODEL.FUSION.NUM_HEADS = 8            # cross_attn 的 head 数
cfg.MODEL.FUSION.DROPOUT = 0.1            # cross_attn 的 dropout











def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


