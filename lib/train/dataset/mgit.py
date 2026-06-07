from __future__ import absolute_import, print_function

from sympy import sequence

from .base_video_dataset import BaseVideoDataset
import glob
import json
import numpy as np
import os
import pandas as pd
import six
import torch
import numpy as np
import pandas
from collections import OrderedDict
from lib.train.data import opencv_loader


class MGIT(BaseVideoDataset):
    r"""`MGIT <http://videocube.aitestunion.com>`_ Dataset.

    Publication:
        ``A Multi-modal Global Instance Tracking Benchmark (MGIT): Better Locating Target in Complex Spatio-temporal and Causal Relationship``, S. Hu, D. Zhang, M. Wu, X. Feng, X. Li, X. Zhao, K. Huang
        Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2023

    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        split (string, optional): Specify ``train``, ``val`` or ``test``
            subset of MGIT.
    """

    def __init__(self, root=None, image_loader=opencv_loader, split=None, version='tiny'):
        super(MGIT, self).__init__('mgit', root, image_loader)
        assert split in ['train', 'val', 'test'], 'Unknown subset.'
        self.base_path = root
        self.split = split

        self.version = version  # temporarily, the toolkit only support tiny version of MGIT

        f = open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'mgit.json'), 'r', encoding='utf-8')
        self.infos = json.load(f)[self.version]
        f.close()

        self.sequence_list = self.infos[self.split]

        if split in ['train', 'val', 'test']:
            self.seq_dirs = [os.path.join(root, 'data', split, s, 'frame_{}'.format(s)) for s in self.sequence_list]
            self.anno_files = [os.path.join(root, 'attribute', 'groundtruth', '{}.txt'.format(s)) for s in
                               self.sequence_list]
            self.restart_files = [os.path.join(root, 'attribute', 'restart', '{}.txt'.format(s)) for s in
                                  self.sequence_list]

    # def get_sequence_info(self, index):
    #     r"""
    #     Args:
    #         index (integer or string): Index or name of a sequence.
    #
    #     Returns:
    #         tuple:
    #             (img_files, anno, restart_flag), where ``img_files`` is a list of
    #             file names, ``anno`` is a N x 4 (rectangles) numpy array, while
    #             ``restart_flag`` is a list of
    #             restart frames.
    #     """
    #     if isinstance(index, six.string_types):
    #         if not index in self.sequence_list:
    #             raise Exception('Sequence {} not found.'.format(index))
    #         index = self.sequence_list.index(index)
    #
    #     img_files = sorted(glob.glob(os.path.join(
    #         self.seq_dirs[index], '*.jpg')))
    #
    #     anno = np.loadtxt(self.anno_files[index], delimiter=',')
    #     nlp_path = '/home/muyh/tracking_datasets/mgit/mgit_nlp/{}.xlsx'.format(
    #         self.sequence_list[index])
    #     nlp_tab = pd.read_excel(nlp_path)
    #     nlp_rect = nlp_tab.iloc[:, [14]].values
    #     nlp_rect = nlp_rect[-1, 0]
    #
    #     # restart_flag = np.loadtxt(self.restart_files[index], delimiter=',', dtype=int)
    #
    #     return img_files, anno, nlp_rect
    #     # return img_files, anno, nlp_rect, restart_flag

    def get_name(self):
        # 可以返回数据集名字或任意标识
        return "mgit"

    def get_frames(self, seq_id, frame_ids, seq_info_dict=None):
        """
        Args:
            seq_id (int): sequence index
            frame_ids (list[int]): frame indices to load
            seq_info_dict (dict or None): returned by get_sequence_info

        Returns:
            frame_list : list of images (as numpy arrays or tensors)
            anno_frames: dict with keys e.g. 'bbox'
            object_meta: OrderedDict with minimal required info
        """
        # 1. 获取序列路径
        seq_path = self.seq_dirs[seq_id]

        # 2. 如果没有传 seq_info_dict，则自己查
        if seq_info_dict is None:
            seq_info_dict = self.get_sequence_info(seq_id)

        bbox_all = seq_info_dict['bbox']  # Nx4 numpy array
        visible_all = seq_info_dict['visible']  # Nx bool
        img_files_all = seq_info_dict['img_files']
        # (如果还有 nlp_rect 等也可以在这里取)
        # ✅ NLP 信息，来自 get_sequence_info 的 nlp_rect
        exp_str = seq_info_dict.get('nlp_rect', None)

        # 3. 逐帧读取图像
        frame_list = []
        for f_id in frame_ids:
            img_path = img_files_all[f_id]
            # 用你的图像读函数，也可换成 PIL 或 opencv
            img = self.image_loader(img_path)
            frame_list.append(img)

        # 4. annotation
        anno_frames = {
            'bbox': [torch.tensor(bbox_all[f_id], dtype=torch.float32) for f_id in frame_ids],
            'visible': [visible_all[f_id] for f_id in frame_ids],
            'nlp':[exp_str] * len(frame_ids)
        }

        # 5. object_meta
        # MGIT 没有 class 名字可以用 sequence_list 自己填
        obj_name = self.sequence_list[seq_id]
        object_meta = OrderedDict({
            'object_class_name': obj_name,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None,
            'exp_str': exp_str  # ✅ ★关键：加入 NLP
        })

        return frame_list, anno_frames, object_meta

    # def get_sequence_info(self, index):
    #     """
    #     Args:
    #         index (int or str): Index or name of a sequence.
    #     Returns:
    #         dict:
    #             'img_files' : list of frame paths
    #             'bbox'      : N x 4 numpy array of bounding boxes
    #             'visible'   : boolean array of valid frames
    #             'nlp_rect'  : NLP rectangle from Excel file
    #     """
    #     # 如果输入是序列名，转换为索引
    #     if isinstance(index, six.string_types):
    #         if index not in self.sequence_list:
    #             raise Exception(f'Sequence {index} not found.')
    #         index = self.sequence_list.index(index)
    #
    #     # 读取图像文件路径
    #     img_files = sorted(glob.glob(os.path.join(self.seq_dirs[index], '*.jpg')))
    #
    #     # 读取标注 bbox
    #     bbox = np.loadtxt(self.anno_files[index], delimiter=',')
    #
    #     # visible 标记：只要宽高都大于 0 就认为有效
    #     visible = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
    #     visible = torch.from_numpy(visible.astype(np.bool_))
    #
    #     # 读取 NLP 文件
    #     nlp_path = os.path.join(
    #         '/home/muyh/tracking_datasets/mgit/mgit_nlp',
    #         f'{self.sequence_list[index]}.xlsx'
    #     )
    #     nlp_tab = pd.read_excel(nlp_path)
    #     nlp_rect = nlp_tab.iloc[-1, 14]  # 取最后一行第15列的值
    #
    #     return {
    #         'img_files': img_files,
    #         'bbox': bbox,
    #         'visible': visible,
    #         'nlp_rect': nlp_rect
    #     }

    def get_sequence_info(self, index):
        """
        Args:
            index (int or str): Index or name of a sequence.
        Returns:
            dict:
                'img_files' : list of frame paths
                'bbox'      : N x 4 numpy array of bounding boxes
                'visible'   : boolean array of valid frames
                'nlp_rect'  : NLP rectangle from Excel file
        """
        # 如果输入是序列名，转换为索引
        if isinstance(index, six.string_types):
            if index not in self.sequence_list:
                raise Exception(f'Sequence {index} not found.')
            index = self.sequence_list.index(index)

        # 读取图像文件路径
        img_files = sorted(glob.glob(os.path.join(self.seq_dirs[index], '*.jpg')))

        # 读取标注 bbox
        bbox = np.loadtxt(self.anno_files[index], delimiter=',')

        # visible 标记：只要宽高都大于 0 就认为有效
        visible = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = torch.from_numpy(visible.astype(np.bool_))

        # 读取 NLP 文件
        nlp_path = os.path.join(
            '/home/muyh/tracking_datasets/mgit/mgit_nlp',
            f'{self.sequence_list[index]}.xlsx'
        )
        nlp_tab = pd.read_excel(nlp_path)
        nlp_rect = nlp_tab.iloc[-1, 14]  # 取最后一行第15列的值

        return {
            'img_files': img_files,
            'bbox': bbox,
            'visible': visible,
            'nlp_rect': nlp_rect
        }



    # def _read_bb_anno(self, seq_id,seq_path):
    #     bb_anno_file = os.path.join(seq_path, 'ground')
    #     bb_anno_file = os.path.join(seq_path, "{seq_id}.txt")
    #     gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
    #     return torch.tensor(gt)
    #
    # def _get_sequence_path(self, seq_id):
    #     seq_name = self.sequence_list[seq_id]
    #     a = 'data/train'
    #     seq_name = os.path.join(a, seq_name)
    #     return os.path.join(self.base_path, seq_name)
    #
    # def _get_sequence_name(self, seq_id):
    #     seq_name = self.sequence_list[seq_id]
    #     return seq_name
    #
    # def get_sequence_info(self, seq_id):
    #     seq_path = self._get_sequence_path(seq_id)
    #     bbox = self._read_bb_anno(seq_id,seq_path)
    #
    #     valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
    #     visible = valid.byte()
    #
    #     return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def __len__(self):
        return len(self.sequence_list)
