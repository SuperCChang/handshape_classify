import logging
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .feature_extract import FEATURE_EXTRACTORS

SIDE_IN_ENGLISH = {'左': 'L', '右': 'R', '左右': 'LR'}

class HandshapeDataset(Dataset):
    def __init__(self, train_path, which_side_path, feature_type="distance_flatten", **extractor_kwargs):
        self.train_path = Path(train_path)
        self.which_side_path = Path(which_side_path)
        
        self.samples = [] 
        self.labels = []  
        
        self._build_dataset(feature_type, extractor_kwargs)

    def _build_dataset(self, feature_type, extractor_kwargs):
        logging.info("读取原始数据目录...")
        df = pd.read_csv(self.which_side_path, encoding='gbk', header=None)
        
        raw_data_list = []
        for _, row in df.iterrows():
            label_id = int(row.iloc[0])
            word_name = str(row.iloc[1]).strip()
            side_info = SIDE_IN_ENGLISH.get(str(row.iloc[2]).strip(), 'LR')
            
            smplx_path = self.train_path / str(label_id) / word_name / 'numpy_result'
            if smplx_path.exists():
                npy_files = sorted(smplx_path.glob('*.npy'))
                npy_list = [np.load(f, allow_pickle=True).item() for f in npy_files]
                raw_data_list.append({
                    'npy_list': npy_list, 
                    'side': side_info, 
                    'label': label_id
                })

        if feature_type not in FEATURE_EXTRACTORS:
            raise ValueError(f"未知的特征提取方法: {feature_type}。可用方法: {list(FEATURE_EXTRACTORS.keys())}")
            
        logging.info(f"使用[{feature_type}]策略进行特征提取...")
        extractor_func = FEATURE_EXTRACTORS[feature_type]
        seq_features, seq_labels = extractor_func(raw_data_list, **extractor_kwargs)

        logging.info("执行序列拆帧处理...")
        for feat_seq, label in zip(seq_features, seq_labels):
            frame_num = feat_seq.shape[0]
            for i in range(frame_num):
                self.samples.append(feat_seq[i].clone())
                self.labels.append(label)

        logging.info(f"数据集构建完毕！共提取出{len(self.samples)}个单帧静态样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
