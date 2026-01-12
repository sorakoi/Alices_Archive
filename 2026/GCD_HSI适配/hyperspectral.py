"""
高光谱数据集加载器 - XCon框架适配版本 (Final Version)

设计原则：
1. 使用13×13空间patch保留空间上下文信息
2. PCA降维到3通道以兼容RGB图像框架
3. 返回标准三元组(img, label, uq_idx)
4. 100%数据用于GCD训练，随机抽50%作为测试参考
5. 完全模仿CIFAR等数据集的接口设计
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import scipy.io as sio
import h5py
from sklearn.decomposition import PCA
from copy import deepcopy
from tqdm import tqdm


# ==========================================
# 数据加载辅助函数（完全不动）
# ==========================================

def load_mat_with_h5py(filename, key):
    """
    使用h5py加载MATLAB v7.3+格式文件

    Args:
        filename: .mat文件路径
        key: 数据键名

    Returns:
        numpy array 或 None（加载失败时）
    """
    try:
        with h5py.File(filename, 'r') as f:
            data = f[key]
            if isinstance(data, h5py.Dataset):
                return np.array(data)
    except Exception as e:
        print(f"H5py加载失败: {e}")
        return None


def load_hsi_mat(dataset_name, data_root):
    """
    加载高光谱.mat文件的统一接口

    支持的数据集配置基于文献中的标准数据集格式

    Args:
        dataset_name: 数据集名称
        data_root: 数据根目录

    Returns:
        data: (H, W, C) float32 array, 原始高光谱数据
        labels: (H, W) int32 array, 地物标签（0为背景）

    Raises:
        ValueError: 未知数据集
        IOError: 文件加载失败
    """
    # 数据集文件配置字典
    # 基于公开数据集的标准格式
    configs = {
        'IndianPines': {
            'data_file': 'Indian_pines_corrected.mat',
            'labels_file': 'Indian_pines_gt.mat',
            'data_key': 'data',
            'labels_key': 'groundT'
        },
        'Pavia': {
            'data_file': 'Pavia.mat',
            'labels_file': 'Pavia_gt.mat',
            'data_key': 'paviaU',
            'labels_key': 'Data_gt'
        },
        'Houston': {
            'data_file': 'Houston13.mat',
            'labels_file': 'Houston13_gt.mat',
            'data_key': 'HSI',
            'labels_key': 'gt'
        },
        'Salinas': {
            'data_file': 'Salinas.mat',
            'labels_file': 'Salinas_gt.mat',
            'data_key': 'HSI_original',
            'labels_key': 'Data_gt'
        },
        'SalinasA': {
            'data_file': 'SalinasA.mat',
            'labels_file': 'SalinasA_gt.mat',
            'data_key': 'salinasA_corrected',
            'labels_key': 'salinasA_gt'
        },
        'Trento': {
            'data_file': 'Trento-HSI.mat',
            'labels_file': 'Trento-GT.mat',
            'data_key': 'HSI',
            'labels_key': 'GT'
        }
    }

    if dataset_name not in configs:
        raise ValueError(f"未知数据集: {dataset_name}. 支持的数据集: {list(configs.keys())}")

    cfg = configs[dataset_name]
    data_path = os.path.join(data_root, cfg['data_file'])
    labels_path = os.path.join(data_root, cfg['labels_file'])

    # 验证文件存在性
    if not os.path.exists(data_path):
        raise IOError(f"数据文件不存在: {data_path}")
    if not os.path.exists(labels_path):
        raise IOError(f"标签文件不存在: {labels_path}")

    # 尝试加载数据
    try:
        data = sio.loadmat(data_path)[cfg['data_key']]
        labels = sio.loadmat(labels_path)[cfg['labels_key']]
    except NotImplementedError:
        # MATLAB v7.3格式需要h5py
        print(f"  检测到MATLAB v7.3格式，使用h5py加载...")
        data = load_mat_with_h5py(data_path, cfg['data_key'])
        labels = load_mat_with_h5py(labels_path, cfg['labels_key'])

        if data is None or labels is None:
            raise IOError(f"无法加载文件，请检查文件格式和键名")

        # h5py可能以(C, H, W)格式返回数据
        if data.ndim == 3 and data.shape[0] < data.shape[1]:
            print(f"  检测到(C,H,W)格式，转置为(H,W,C)")
            data = np.transpose(data, (1, 2, 0))

    return data.astype(np.float32), labels.astype(np.int32)


# ==========================================
# 数据预处理函数（完全不动）
# ==========================================

def normalize_hsi(data):
    """
    逐波段Min-Max归一化到[0, 1]

    采用逐波段归一化策略，保留各波段的相对强度关系。
    添加epsilon避免数值不稳定性。

    Args:
        data: (H, W, C) numpy array

    Returns:
        normalized: (H, W, C) numpy array, 值域[0, 1]
    """
    h, w, c = data.shape
    data_flat = data.reshape(-1, c)

    # 逐波段计算最小值和最大值
    data_min = np.min(data_flat, axis=0, keepdims=True)  # (1, C)
    data_max = np.max(data_flat, axis=0, keepdims=True)  # (1, C)
    data_range = data_max - data_min

    # 防止除零
    epsilon = 1e-8
    data_scaled = (data_flat - data_min) / (data_range + epsilon)

    return data_scaled.reshape(h, w, c)


def reduce_to_rgb_pca(data, n_components=3, random_state=42):
    """
    使用PCA将高维光谱数据降维到3通道

    PCA能够保留主要的光谱变异信息，同时将数据压缩到
    RGB图像框架兼容的3通道格式。

    设计考量：
    - n_components=3: 兼容标准RGB图像处理流程
    - 降维后再次归一化: 确保值域在[0,1]以便转换为图像

    Args:
        data: (H, W, C) numpy array, 归一化后的HSI数据
        n_components: 目标通道数，默认3
        random_state: PCA随机种子

    Returns:
        rgb_data: (H, W, 3) numpy array, 值域[0, 1]
    """
    h, w, c = data.shape
    data_flat = data.reshape(-1, c)

    pca = PCA(n_components=n_components, random_state=random_state)
    rgb_flat = pca.fit_transform(data_flat)

    # 计算保留的方差比例
    variance_ratio = pca.explained_variance_ratio_.sum()
    print(f"  PCA降维: {c} → {n_components}通道, 保留方差: {variance_ratio * 100:.2f}%")

    # 重要：PCA后的值域可能改变，需要重新归一化到[0,1]
    rgb_min = rgb_flat.min()
    rgb_max = rgb_flat.max()
    rgb_flat = (rgb_flat - rgb_min) / (rgb_max - rgb_min + 1e-8)

    return rgb_flat.reshape(h, w, n_components)


# ==========================================
# 核心Dataset类（修改split逻辑）
# ==========================================

class HSIDataset(Dataset):
    """
    高光谱数据集 - 完全模仿CIFAR结构

    关键修改：
    - split='all': 返回100%数据（用于GCD训练）
    - split='test_sample': 返回随机抽样的50%数据（测试参考，允许重叠）
    """

    def __init__(self, root, dataset_name, split='all',
                 transform=None, target_transform=None,
                 patch_size=13, test_sample_ratio=0.5, seed=42):
        """
        初始化HSI数据集

        Args:
            root: 数据根目录
            dataset_name: 数据集名称
            split: 'all' 或 'test_sample'
            transform: 外部传入的transform
            target_transform: 外部传入的target_transform
            patch_size: 空间patch大小，默认13
            test_sample_ratio: 测试抽样比例，默认0.5
            seed: 随机种子，确保可复现
        """

        # 基础属性
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.patch_size = patch_size
        self.seed = seed

        print(f"\n{'=' * 60}")
        print(f"初始化HSIDataset: {dataset_name} (split={split})")
        print(f"{'=' * 60}")

        # ==========================================
        # 步骤1: 加载原始数据
        # ==========================================
        data, labels = load_hsi_mat(dataset_name, root)

        print(f"✓ 原始数据加载完成")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  波段数: {data.shape[2]}")

        # ==========================================
        # 步骤2: 数据预处理
        # ==========================================
        print(f"\n正在预处理数据...")

        # 归一化
        data = normalize_hsi(data)
        print(f"✓ 归一化完成")

        # PCA降维到3通道
        data = reduce_to_rgb_pca(data, n_components=3, random_state=seed)
        print(f"✓ PCA降维完成，shape: {data.shape}")

        # ==========================================
        # 步骤3: 填充数据以便提取patch
        # ==========================================
        pad = self.patch_size // 2
        self.padded_data = np.pad(
            data,
            ((pad, pad), (pad, pad), (0, 0)),
            mode='reflect'
        )
        print(f"✓ 数据填充完成")

        # ==========================================
        # 步骤4: 提取所有前景像素（100%）
        # ==========================================
        print(f"\n正在提取前景像素...")

        all_coords = []
        all_labels_raw = []

        h, w = labels.shape

        for i in range(h):
            for j in range(w):
                if labels[i, j] != 0:  # 0是背景
                    all_coords.append((i, j))
                    all_labels_raw.append(int(labels[i, j]))

        print(f"✓ 找到 {len(all_coords)} 个前景像素")

        # ==========================================
        # 步骤5: 创建标签映射
        # ==========================================
        unique_labels = sorted(list(set(all_labels_raw)))
        self.label_mapping = {orig: idx for idx, orig in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

        # 应用标签映射
        all_labels_mapped = [self.label_mapping[lbl] for lbl in all_labels_raw]

        print(f"\n标签映射:")
        print(f"  原始标签: {unique_labels}")
        print(f"  类别数: {self.num_classes}")

        # ==========================================
        # 步骤6: 根据split决定返回哪些数据
        # ==========================================
        if split == 'all':
            # 返回100%数据
            coords_list = all_coords
            labels_list = all_labels_mapped
            print(f"\n✓ 使用100%数据 ({len(coords_list)} 样本)")

        elif split == 'test_sample':
            # 随机抽样作为测试参考（允许和训练集重叠）
            np.random.seed(seed)
            n_samples = len(all_coords)
            n_test = int(n_samples * test_sample_ratio)

            test_indices = np.random.choice(n_samples, n_test, replace=False)

            coords_list = [all_coords[i] for i in test_indices]
            labels_list = [all_labels_mapped[i] for i in test_indices]
            print(f"\n✓ 随机抽样{test_sample_ratio*100:.0f}%数据作为测试参考")
            print(f"  测试样本数: {len(coords_list)}")
            print(f"  注: 测试集可能与训练集重叠，仅供参考")

        else:
            raise ValueError(f"未知split: {split}，应为'all'或'test_sample'")

        # ==========================================
        # 步骤7: 创建samples（模仿CIFAR）
        # ==========================================
        self.samples = [
            (coord, label)
            for coord, label in zip(coords_list, labels_list)
        ]

        self.uq_idxs = np.array(range(len(self.samples)))

        print(f"\n{'=' * 60}")
        print(f"数据集初始化完成 (split={split})")
        print(f"  样本数: {len(self.samples)}")
        print(f"  Patch大小: {patch_size}×{patch_size}")
        print(f"{'=' * 60}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        完全模仿Flowers102的访问模式
        """
        # 解包坐标和标签
        coord, label = self.samples[idx]
        y, x = coord

        # 提取空间patch
        patch = self.padded_data[y:y + self.patch_size, x:x + self.patch_size, :]

        # 转换为PIL Image
        patch_uint8 = (patch * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(patch_uint8, mode='RGB')

        # 应用transform
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, self.uq_idxs[idx]


# ==========================================
# 数据集操作工具函数（完全不动）
# ==========================================

def subsample_dataset(dataset, idxs):
    """完全模仿Flowers102"""
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = [(c, l) for i, (c, l) in enumerate(dataset.samples) if i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes):
    """完全模仿Flowers102"""
    cls_idxs = [i for i, (_, label) in enumerate(dataset.samples) if label in include_classes]

    target_xform_dict = {k: i for i, k in enumerate(include_classes)}

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


# ==========================================
# get_datasets接口函数（修改数据流）
# ==========================================

def get_hsi_datasets(dataset_name, train_transform, test_transform,
                     train_classes, prop_train_labels=0.1,
                     split_train_val=False, seed=42):
    """
    获取HSI数据集，完全遵循GCD框架的标准接口

    修改逻辑：
    - 100%数据用于训练（train_labelled + train_unlabelled）
    - 随机抽50%作为测试参考（test_dataset，允许重叠）
    """
    np.random.seed(seed)

    print(f"\n{'=' * 70}")
    print(f"创建HSI数据集: {dataset_name}")
    print(f"{'=' * 70}")
    print(f"配置:")
    print(f"  已知类: {list(train_classes)}")
    print(f"  有标签数据比例: {prop_train_labels}")
    print(f"  随机种子: {seed}")
    print(f"{'=' * 70}\n")

    # ==========================================
    # 步骤1: 加载完整数据集（100%用于训练）
    # ==========================================
    print("【步骤1】加载完整数据集（100%）...")
    whole_training_set = HSIDataset(
        root='data',
        dataset_name=dataset_name,
        split='all',  # 修改：使用100%数据
        transform=train_transform,
        seed=seed
    )
    print(f"✓ 完整数据集: {len(whole_training_set)} 样本")
    print(f"✓ 类别数: {whole_training_set.num_classes}")

    # ==========================================
    # 步骤2: 从已知类中创建D_L
    # ==========================================
    print("【步骤2】筛选已知类并创建D_L...")
    train_dataset_labelled = subsample_classes(
        deepcopy(whole_training_set),
        include_classes=train_classes
    )
    print(f"✓ 已知类筛选后: {len(train_dataset_labelled)} 样本")

    # 按比例采样
    from data.data_utils import subsample_instances
    subsample_indices = subsample_instances(
        train_dataset_labelled,
        prop_indices_to_subsample=prop_train_labels
    )
    train_dataset_labelled = subsample_dataset(
        train_dataset_labelled,
        subsample_indices
    )
    print(f"✓ D_L (有标签): {len(train_dataset_labelled)} 样本")

    # ==========================================
    # 步骤3: 创建D_U（剩余的所有数据）
    # ==========================================
    print("\n【步骤3】创建D_U（unlabelled数据）...")
    unlabelled_indices = (set(whole_training_set.uq_idxs) -
                          set(train_dataset_labelled.uq_idxs))
    train_dataset_unlabelled = subsample_dataset(
        deepcopy(whole_training_set),
        np.array(list(unlabelled_indices))
    )
    print(f"✓ D_U (无标签): {len(train_dataset_unlabelled)} 样本")
    print(f"  包含: 已知类剩余样本 + 所有未知类样本")

    # ==========================================
    # 步骤4: 创建测试参考集
    # ==========================================
    print("\n【步骤4】创建测试集（使用unlabelled数据）...")
    test_dataset = deepcopy(train_dataset_unlabelled)
    test_dataset.transform = test_transform  # 应用测试时的transform
    print(f"✓ 测试集: {len(test_dataset)} 样本")
    print(f"  注: 测试集与train_unlabelled完全相同（仅transform不同）")

    # ==========================================
    # 验证数据一致性
    # ==========================================
    print(f"\n{'=' * 70}")
    print(f"数据集创建完成 - 统计信息")
    print(f"{'=' * 70}")
    print(f"训练数据（100%参与训练）:")
    print(f"  ├─ D_L (有标签):          {len(train_dataset_labelled):6d} 样本")
    print(f"  └─ D_U (无标签):          {len(train_dataset_unlabelled):6d} 样本")
    print(f"  ─────────────────────────────────────")
    print(f"     训练集总计:            {len(whole_training_set):6d} 样本")
    print(f"")
    print(f"测试数据:")
    print(f"  └─ 测试集 (=D_U):         {len(test_dataset):6d} 样本")
    print(f"")
    print(f"类别信息:")
    print(f"  ├─ 已知类数量:            {len(train_classes)}")
    print(f"  ├─ 未知类数量:            {whole_training_set.num_classes - len(train_classes)}")
    print(f"  └─ 总类别数:              {whole_training_set.num_classes}")
    print(f"")
    print(f"评估说明:")
    print(f"  ● 训练: 使用train_labelled + train_unlabelled")
    print(f"  ● 测试: 在test_dataset上评估（与train_unlabelled相同但transform不同）")
    print(f"{'=' * 70}\n")


    return {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': None,
        'test': test_dataset
    }


# ==========================================
# 各数据集的包装函数（添加seed参数）
# ==========================================

def get_indianpines_datasets(train_transform, test_transform, train_classes,
                             prop_train_labels=0.1, split_train_val=False, seed=42):
    return get_hsi_datasets('IndianPines', train_transform, test_transform,
                            train_classes, prop_train_labels, split_train_val, seed)


def get_pavia_datasets(train_transform, test_transform, train_classes,
                       prop_train_labels=0.1, split_train_val=False, seed=42):
    return get_hsi_datasets('Pavia', train_transform, test_transform,
                            train_classes, prop_train_labels, split_train_val, seed)


def get_houston_datasets(train_transform, test_transform, train_classes,
                         prop_train_labels=0.1, split_train_val=False, seed=42):
    return get_hsi_datasets('Houston', train_transform, test_transform,
                            train_classes, prop_train_labels, split_train_val, seed)


def get_salinas_datasets(train_transform, test_transform, train_classes,
                         prop_train_labels=0.1, split_train_val=False, seed=42):
    return get_hsi_datasets('Salinas', train_transform, test_transform,
                            train_classes, prop_train_labels, split_train_val, seed)


def get_salinasa_datasets(train_transform, test_transform, train_classes,
                          prop_train_labels=0.1, split_train_val=False, seed=42):
    return get_hsi_datasets('SalinasA', train_transform, test_transform,
                            train_classes, prop_train_labels, split_train_val, seed)


def get_trento_datasets(train_transform, test_transform, train_classes,
                        prop_train_labels=0.1, split_train_val=False, seed=42):
    return get_hsi_datasets('Trento', train_transform, test_transform,
                            train_classes, prop_train_labels, split_train_val, seed)
