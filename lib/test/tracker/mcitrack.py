
from pytorch_pretrained_bert import BertTokenizer

from lib.models.language_model import build_bert
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.test.utils.hann import hann2d
from lib.models.mcitrack import build_mcitrack
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np
import os
import math
import csv

from lib.utils.misc import NestedTensor


class MCITRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MCITRACK, self).__init__(params)
        network = build_mcitrack(params.cfg)
        network.load_state_dict(
            torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        # strict=False: consistency 模块无可学习参数不会产生 missing keys，
        # 但保留 False 以兼容原始代码中已存在的非 strict 模块 (如 text_proj)
        vocab_path = '/home/zhanggt/pretrained/bert/bert-base-uncased-vocab.txt'
        self.k = 1
        if vocab_path is not None and os.path.exists(vocab_path):
            self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.params.cfg.MODEL.LANGUAGE.TYPE, do_lower_case=True)
        self.language_backbone = build_bert().cuda()
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True:  # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0
        # for update
        self.h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS
        if self.debug == 2 :
            save_dir = "/home/kb/kb/MCITrack/vis"
            self.save_dir = os.path.join(save_dir, params.yaml_name)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPT, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPT[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPT.DEFAULT
        print("Update threshold is: ", self.update_threshold)

        if hasattr(self.cfg.TEST.UPH, DATASET_NAME):
            self.update_h_t = self.cfg.TEST.UPH[DATASET_NAME]
        else:
            self.update_h_t = self.cfg.TEST.UPH.DEFAULT
        print("Update hidden state threshold is: ", self.update_h_t)

        if hasattr(self.cfg.TEST.INTER, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.INTER[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.INTER.DEFAULT
        print("Update intervals is: ", self.update_intervals)

        if hasattr(self.cfg.TEST.MB, DATASET_NAME):
            self.memory_bank = self.cfg.TEST.MB[DATASET_NAME]
        else:
            self.memory_bank = self.cfg.TEST.MB.DEFAULT
        print("Update threshold is: ", self.memory_bank)

        # === 方向 C: 一致性模块配置 ===
        # P0 (默认 False): consistency 仅诊断输出, 不影响任何决策
        # P1 (True): 仅用于 h_state reset, 不影响模板更新/淘汰
        self.use_consistency_for_h_reset = getattr(
            self.cfg.TEST, 'USE_CONSISTENCY_FOR_H_RESET', False)
        self.consistency_reset_alpha = getattr(
            self.cfg.TEST, 'CONSISTENCY_RESET_ALPHA', 0.5)
        print("Use consistency for h_reset: ", self.use_consistency_for_h_reset)
        print("Consistency reset alpha: ", self.consistency_reset_alpha)

        # [修复 ②] consistency 在线标定: 用序列内 EMA 均值/方差做 z-score + sigmoid，
        # 把 consistency 映射到与 conf_score 可比的 (0,1) 尺度后再线性混合。
        self.consistency_calib_momentum = getattr(
            self.cfg.TEST, 'CONSISTENCY_CALIB_MOMENTUM', 0.95)
        self._cons_ema_mean = None
        self._cons_ema_var = None

        # === 诊断日志开关 (默认关; 开启后每个序列写一个 CSV, 不影响跟踪结果) ===
        self.log_diagnostics = getattr(self.cfg.TEST, 'LOG_DIAGNOSTICS', False)
        self.diag_dir = getattr(self.cfg.TEST, 'DIAG_DIR', './diag_logs')
        self.seq_idx = -1
        self.diag_file = None
        print("Log diagnostics: ", self.log_diagnostics, "-> dir:", self.diag_dir)

    def initialize(self, image, info: dict):
        if self.debug == 2:
            self.save_path = os.path.join(self.save_dir, info['seq_name'])
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                   output_sz=self.params.template_size)
        z_patch_arr = z_patch_arr


        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        text_input = self._text_input_process(info['init_nlp'], 30)
        self.text_input = self.language_backbone(text_input).to(template.device)

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)] * self.num_template
        self.frame_id = 0
        self.memory_template_list = self.template_list.copy()
        self.memory_template_anno_list = self.template_anno_list.copy()

        # [修复 ②] 每个序列独立重置一致性在线标定统计
        self._cons_ema_mean = None
        self._cons_ema_var = None
        self._frames_since_h_reset = 0  # 距上次 h_state 重置的帧数 (记忆"年龄")

        # === 诊断日志: 每个序列一个 CSV, 写表头 ===
        if self.log_diagnostics:
            self.seq_idx += 1
            seq_name = info.get('seq_name', None) if isinstance(info, dict) else None
            seq_name = seq_name if seq_name else ("seq_%04d" % self.seq_idx)
            safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(seq_name))
            os.makedirs(self.diag_dir, exist_ok=True)
            self.diag_file = os.path.join(self.diag_dir, safe + ".csv")
            with open(self.diag_file, "w", newline="") as f:
                csv.writer(f).writerow([
                    "frame_id", "conf_score", "consistency", "calibrated_consistency",
                    "combined_score", "apce", "v_per_template",
                    "h_reset_by_conf", "h_reset_by_combined", "final_h_reset", "h_age",
                    "memory_added", "clip_refreshed", "memory_len", "num_template",
                    "x", "y", "w", "h"])

    def _calibrate_consistency(self, c):
        """
        [修复 ②] 序列内在线 z-score + sigmoid 标定，输出 ∈ (0,1)，与 conf_score 同尺度。
        含义: 本帧一致性相对该序列 "典型水平" 是偏高还是偏低。
        首帧无统计 -> 返回中性 0.5。用更新前的统计量算 z，再更新 EMA (避免自我抵消)。
        """
        m = self.consistency_calib_momentum
        if self._cons_ema_mean is None:
            self._cons_ema_mean = c
            self._cons_ema_var = 1e-4
            return 0.5
        std = max(math.sqrt(self._cons_ema_var) if self._cons_ema_var > 1e-8 else 1e-3, 1e-3)
        z = (c - self._cons_ema_mean) / std
        delta = c - self._cons_ema_mean
        self._cons_ema_mean = m * self._cons_ema_mean + (1.0 - m) * c
        self._cons_ema_var = m * self._cons_ema_var + (1.0 - m) * (delta * delta)
        return 1.0 / (1.0 + math.exp(-z))

    def _log_frame(self, conf, cons, cal_cons, combined, apce, v_per_template,
                   h_conf, h_comb, h_final, h_age, mem_added, clip_refreshed):
        """[诊断] 把本帧关键量追加写入序列 CSV。日志失败绝不影响跟踪。"""
        try:
            x, y, w, h = self.state
            v_str = ";".join("%.4f" % v for v in (v_per_template or []))
            with open(self.diag_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.frame_id, "%.4f" % conf, "%.4f" % cons, "%.4f" % cal_cons,
                    "%.4f" % combined, "%.4f" % apce, v_str,
                    int(bool(h_conf)), int(bool(h_comb)), int(bool(h_final)), int(h_age),
                    int(bool(mem_added)), int(bool(clip_refreshed)),
                    len(self.memory_template_list), self.num_template,
                    round(float(x), 2), round(float(y), 2),
                    round(float(w), 2), round(float(h), 2)])
        except Exception:
            pass

    def _text_input_process(self, nlp, seq_length):
        text_ids, text_masks = self._extract_token_from_nlp(nlp, seq_length)
        text_ids = torch.tensor(text_ids).unsqueeze(0).cuda()
        text_masks = torch.tensor(text_masks).unsqueeze(0).cuda()
        return NestedTensor(text_ids, text_masks)

    def _extract_token_from_nlp(self, nlp, seq_length):
        """ use tokenizer to convert nlp to tokens
        param:
            nlp:  a sentence of natural language
            seq_length: the max token length, if token length larger than seq_len then cut it,
            elif less than, append '0' token at the reef.
        return:
            token_ids and token_marks
        """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]
        # build tokens and token_ids
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return input_ids, input_mask

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        search_list = [search]

        # run the encoder
        with torch.no_grad():
            enc_opt = self.network.forward_encoder(self.template_list, search_list, self.template_anno_list)

        # === S1 fix: 在 Neck 之前计算一致性分数 ===
        # enc_opt 中的模板 tokens 尚未被 Neck 的 ViT self-attention 混合
        # Token 顺序: [search(L_x) | template_0(L_z) | ... | template_{N-1}(L_z)]
        with torch.no_grad():
            search_feat_raw = enc_opt[:, 0:self.network.num_patch_x]  # (B, L_x, D)
            V_scores = self.network.forward_consistency(enc_opt, search_feat_raw)  # (B, N)
            avg_consistency = V_scores.mean().item()  # 诊断用，不用于门控
            v_per_template = (V_scores.detach().squeeze(0).cpu().tolist()
                              if self.log_diagnostics else None)  # 每个模板的可靠度

        # run the time neck
        with torch.no_grad():
            hidden_state = self.h_state.copy()
            encoder_out, out_neck, h = self.network.forward_neck(enc_opt, hidden_state)

        # === TemplateAwareFusion (PMIFM 迁移) ===
        # enc_opt 在 forward_neck 后仍是原始 Encoder 输出 (无 in-place op)
        # FUSION.ENABLE=False 时 forward_fusion 直接返回 out_neck
        with torch.no_grad():
            out_neck = self.network.forward_fusion(enc_opt, out_neck)

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.forward_decoder(feature=out_neck)

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True:  # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map

        # [诊断] APCE: 响应图锐度, 经典的独立可靠度指标 (与 conf/consistency 对照用)
        apce = 0.0
        if self.log_diagnostics:
            with torch.no_grad():
                _r = response.detach().reshape(-1).float()
                _fmin = _r.min(); _fmax = _r.max()
                apce = ((_fmax - _fmin) ** 2 / (((_r - _fmin) ** 2).mean() + 1e-8)).item()
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'],
                                                                   out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response,
                                                                   out_dict['offset_map'],
                                                                   return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # ====================================================================
        # h_state 重置逻辑 (P0 / P1 双路径)
        # --------------------------------------------------------------------
        # P0 (USE_CONSISTENCY_FOR_H_RESET=False, 默认):
        #   仅用 conf_score 触发, 与 baseline 完全一致
        #   consistency 仅在 return dict 中输出, 不影响 tracking
        # P1 (USE_CONSISTENCY_FOR_H_RESET=True):
        #   用 alpha*conf + (1-alpha)*consistency 联合触发 h_state reset
        #   仅改变 h_state reset 一个决策点, 用于隔离 ablation
        #   不影响模板更新、模板淘汰、bbox 预测等其他决策
        # ====================================================================
        conf_score_val = conf_score.item() if hasattr(conf_score, 'item') else float(conf_score)
        alpha = self.consistency_reset_alpha
        # [修复 ②] 先把 consistency 标定到与 conf 可比的 (0,1) 尺度, 再线性混合
        calibrated_consistency = self._calibrate_consistency(avg_consistency)
        combined_score = alpha * conf_score_val + (1.0 - alpha) * calibrated_consistency

        # 计算诊断标志 (无论开关状态, 都计算用于 return dict 分析)
        h_reset_by_conf = conf_score_val < self.update_h_t
        h_reset_by_combined = combined_score < self.update_h_t

        self.h_state = h
        if self.use_consistency_for_h_reset:
            # P1: combined_score 触发 reset
            final_h_reset = h_reset_by_combined
        else:
            # P0/baseline: 仅 conf_score 触发 reset
            final_h_reset = h_reset_by_conf

        if final_h_reset:
            self.h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS
            self._frames_since_h_reset = 0
        else:
            self._frames_since_h_reset += 1

        # === 模板更新: 仅用 conf_score 门控 (baseline, P0/P1 均不变) ===
        # consistency 不参与模板更新决策
        memory_added = False
        if self.num_template > 1:
            if (conf_score > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.memory_template_list.append(template)
                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.memory_template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                memory_added = True
                # === FIFO 淘汰 (baseline, P0/P1 均不变) ===
                if len(self.memory_template_list) > self.memory_bank:
                    self.memory_template_list.pop(0)
                    self.memory_template_anno_list.pop(0)
        clip_refreshed = False
        if (self.frame_id % self.update_intervals == 0):
            assert len(self.memory_template_anno_list) == len(self.memory_template_list)
            len_list = len(self.memory_template_anno_list)
            interval = len_list // self.num_template
            for i in range(1, self.num_template):
                idx = interval * i
                if idx > len_list:
                    idx = len_list
                self.template_list.append(self.memory_template_list[idx])
                self.template_list.pop(1)
                self.template_anno_list.append(self.memory_template_anno_list[idx])
                self.template_anno_list.pop(1)
            clip_refreshed = True
        assert len(self.template_list) == self.num_template

        # === 诊断日志: 记录本帧 conf / consistency / 模板替换等 ===
        if self.log_diagnostics and self.diag_file is not None:
            self._log_frame(conf_score_val, avg_consistency, calibrated_consistency,
                            combined_score, apce, v_per_template,
                            h_reset_by_conf, h_reset_by_combined, final_h_reset,
                            self._frames_since_h_reset, memory_added, clip_refreshed)

        # for debug
        if self.debug == 2:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_path, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        elif self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state,
                "best_score": conf_score,
                "conf_score": conf_score_val,
                "consistency": avg_consistency,
                "calibrated_consistency": calibrated_consistency,
                "combined_score": combined_score,
                "h_reset_by_conf": h_reset_by_conf,
                "h_reset_by_combined": h_reset_by_combined,
                "final_h_reset": final_h_reset,
                "use_consistency_for_h_reset": self.use_consistency_for_h_reset,
                "consistency_reset_alpha": self.consistency_reset_alpha}


    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return MCITRACK

