
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

        conf_score_val = conf_score.item() if hasattr(conf_score, 'item') else float(conf_score)

        # h_state 重置 (baseline: 仅 conf_score 触发)
        self.h_state = h
        if conf_score_val < self.update_h_t:
            self.h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS

        # === 模板更新: 仅用 conf_score 门控 (baseline) ===
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
                # === FIFO 淘汰 (baseline) ===
                if len(self.memory_template_list) > self.memory_bank:
                    self.memory_template_list.pop(0)
                    self.memory_template_anno_list.pop(0)
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
        assert len(self.template_list) == self.num_template

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
                "best_score": conf_score}


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

