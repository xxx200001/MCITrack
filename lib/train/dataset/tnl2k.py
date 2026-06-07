import json
import os
import os.path
import random
import re
from collections import OrderedDict

# import ipdb
import numpy as np
import pandas
import torch

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset


class Tnl2k(BaseVideoDataset):
    """ TNL2K dataset.

    Publication:
        Towards More Flexible and Accurate Object Tracking with Natural Language: Algorithms and Benchmark
        Xiao Wang, Xiujun Shu, Zhipeng Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, Feng Wu CVPR 2021 2021

    Download the dataset from https://sites.google.com/view/langtrackbenchmark/
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None,
                 multi_modal_vision=False, multi_modal_language=False):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root = root + '/' + split
        super().__init__('tnl2k', self.root, image_loader)

        # Keep a list of all classes
        self.sequence_list = self._build_sequence_list()

        self.data_index = self._create_data_index()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.multi_modal_language = multi_modal_language

    def _build_sequence_list(self):
        # todo update split
        sequence_list = []
        subset_list = [f for f in os.listdir(self.root)
                       if os.path.isdir(os.path.join(self.root, f)) and f != 'revised_annotations']

        # one-level directory: ['INF_womanleft', 'INF_whitesuv', ...]
        if len(subset_list) > 14:
            self.dir_type = 'one-level'
            return sorted(subset_list)

        # two-level directory: ['TNL2k_train_subset_p9/INF_womanleft', 'TNL2k_train_subset_p9/INF_whitesuv', ...]
        self.dir_type = 'two-level'
        for x in subset_list:
            sub_sequence_list_path = os.path.join(self.root, x)
            for seq in os.listdir(sub_sequence_list_path):
                sequence_list.append(os.path.join(x, seq))
        sequence_list = sorted(sequence_list)
        return sequence_list

    def _create_data_index(self):
        tnl2k_cache_root = os.path.join(os.path.expanduser('~'), '.cache', 'tnl2k')
        if not os.path.exists(tnl2k_cache_root):
            os.makedirs(tnl2k_cache_root)
        if os.path.exists(os.path.join(tnl2k_cache_root, 'index.json')):
            with open(os.path.join(tnl2k_cache_root, 'index.json'), "r") as f:
                tnl2k_index = json.load(f)
        else:
            tnl2k_index = {}
            print("saving index for tnl2k...")
            for seq in self.sequence_list:
                img_list = os.listdir(os.path.join(self.root, seq, 'imgs'))
                img_list = self._sort(img_list)
                tnl2k_index[seq] = img_list
            with open(os.path.join(tnl2k_cache_root, 'index.json'), "w") as f:
                json.dump(tnl2k_index, f)

        return tnl2k_index

    def _sort(self, img_list):
        """对包含数字的文件名进行正确的数值排序"""
        import re
        img_list.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
        return img_list

    def _build_class_list(self):
        # if not self.has_class_info():
        return None

    def get_name(self):
        return 'tnl2k'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return None

    def get_sequences_in_class(self, class_name):
        if self.has_class_info():
            return self.seq_per_class[class_name]
        else:
            return None

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    # def _read_nlp(self, seq_path):
    #     nlp_file = os.path.join(seq_path, "language.txt")
    #     nlp = ""
    #     try:
    #         nlp = pandas.read_csv(nlp_file, dtype=str, header=None, low_memory=False).values
    #     except Exception as e:
    #         print(e)
    #         print(f'nlp_file:{nlp_file}')
    #     return nlp[0][0]

    def _read_nlp(self, seq_path):
        nlp_file = os.path.join(seq_path, "language.txt")
        try:
            # 正常读取
            nlp = pandas.read_csv(nlp_file, dtype=str, header=None, low_memory=False).values
            return str(nlp[0][0])
        except Exception as e:
            # 遇到空文件或损坏文件时的兜底策略 (Fallback)
            # print(f"Warning: Empty or missing language file: {nlp_file}. Using fallback.")

            # 提取类别名作为兜底的自然语言提示 (比如 'INF_womanpink' -> 提取出 'INF_womanpink' 或 'woman')
            fallback_text = self._get_class(seq_path)

            # 如果连类别名都提取失败，就用最通用的 "target"
            if not fallback_text:
                fallback_text = "target"

            return fallback_text

    def _read_target_visible(self, seq_path):
        # Read groundtruth.txt
        bbox = self._read_bb_anno(seq_path)

        target_visible = (bbox[:, 0] > 0) | (bbox[:, 1] > 0) | (bbox[:, 2] > 0) | (bbox[:, 3] > 0)

        return target_visible

    def _get_sequence_path(self, seq_id):
        if self.dir_type == 'two-level':
            seq_name = self.sequence_list[seq_id].split('/')[-1]
            class_name = self.sequence_list[seq_id].split('/')[0]
            return os.path.join(self.root, class_name, seq_name)
        else:
            seq_name = self.sequence_list[seq_id]
            return os.path.join(self.root, seq_name)

    def _get_sequence_name(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return seq_name

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()
        output = {'bbox': bbox, 'valid': valid, 'visible': visible}
        if self.multi_modal_language:
            nlp = self._read_nlp(seq_path)
            output['nlp'] = nlp
        return output

    def get_sequence_nlp(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        nlp = self._read_nlp(seq_path)
        return nlp

    # def _get_frame_path(self, seq_path, frame_id):
    #     # TNL2K Not all frame start from 1 and the images name were not unified
    #     path_list = os.listdir(os.path.join(seq_path, 'imgs'))
    #     regex_end = re.compile(r"[0-9]*")
    #     try:
    #         path_list.sort(key=lambda x: int(re.findall(regex_end, x)[0]))
    #     except:
    #         raise ValueError("worng change str to int")
    #     return os.path.join(seq_path, 'imgs', path_list[frame_id - 1])

    # def _get_frame_path(self, seq_path, frame_id):
    #     images_path = sorted(os.listdir(os.path.join(seq_path, 'imgs')))
    #     image_path = os.path.join(seq_path, 'imgs', images_path[frame_id])
    #     return image_path

    def _get_frame_path(self, seq_path, seq_name, frame_id):
        # should speed up
        # img_list = os.listdir(os.path.join(seq_path, 'imgs'))
        # img_list = self._sort(img_list)
        img_list = self.data_index[seq_name]
        try:
            img_name = img_list[frame_id]
        except Exception as e:
            print('ERROR: Could not find image "{}"'.format(os.path.join(seq_path, 'frames', img_name)))
            print(e)
            return None

        return os.path.join(seq_path, 'imgs', img_name)

    def _get_frame(self, seq_path, seq_name, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, seq_name, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        seq_name = self._get_sequence_name(seq_id)
        obj_class = self._get_class(seq_path)

        frame_list = [self._get_frame(seq_path, seq_name, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = [value for _ in frame_ids]
            else:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        # exp_str = self.get_sequence_nlp(seq_id)

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None
                                   })

        return frame_list, anno_frames, object_meta

    def get_path(self, seq_id, frame_ids):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame_path(seq_path, f_id) for f_id in frame_ids]
        return frame_list
