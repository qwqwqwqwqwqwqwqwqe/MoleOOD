# 文件: models/utils.py

import torch
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score

def split_into_groups(g):
    unique_groups, unique_counts = torch.unique(
        g, sorted=False, return_counts=True
    )
    group_indices = [
        torch.nonzero(g == group, as_tuple=True)[0]
        for group in unique_groups
    ]
    return unique_groups, group_indices, unique_counts

def pyg_batch_from_subgraphs(subgraphs_list_of_lists):
    """
    将一个批次中所有分子的子结构图列表，
    重新拼接成一个巨大的 PyG Batch 对象。
    
    输入: [[g1, g2], [g3], [g4, g5, g6], ...]  (List of Lists of Data objects)
    输出: 一个 Batch 对象，包含了所有 g1...g6 的信息
    """
    # 1. 展平列表：把列表的列表变成一个长长的一维列表
    flat_subgraphs = [sub for sub_list in subgraphs_list_of_lists for sub in sub_list]
    
    # 2. 如果没有任何子结构，返回 None
    if not flat_subgraphs:
        return None
        
    # 3. 使用 PyG 的 Batch.from_data_list() 魔法进行拼接
    return Batch.from_data_list(flat_subgraphs)

def get_subgraph_batch_index(subgraphs_list_of_lists):
    """
    创建一个索引，用于在注意力机制中，将子结构特征映射回它们所属的原始分子。
    """
    indices = []
    for i, sub_list in enumerate(subgraphs_list_of_lists):
        indices.extend([i] * len(sub_list))
    
    if not indices:
        return torch.tensor([], dtype=torch.long)
        
    return torch.tensor(indices, dtype=torch.long)
  
def evaluate(pred, gt, metric='auc'):
    if isinstance(metric, str):
        metric = [metric]
    allowed_metric = ['auc', 'accuracy']
    invalid_metric = set(metric) - set(allowed_metric)
    if len(invalid_metric) != 0:
        raise ValueError(f'Invalid Value {invalid_metric}')
    result = {}
    for M in metric:
        if M == 'auc':
            all_prob = pred[:, 0] + pred[:, 1]
            assert torch.all(torch.abs(all_prob - 1) < 1e-2), \
                "Input should be a binary distribution"
            score = pred[:, 1]
            result[M] = roc_auc_score(gt, score)
        else:
            pred = pred.argmax(dim=-1)
            total, correct = len(pred), torch.sum(pred.long() == gt.long())
            result[M] = (correct / total).item()
    return result