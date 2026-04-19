import torch
import math
from torch_geometric.nn import global_mean_pool
from .utils import pyg_batch_from_subgraphs, get_subgraph_batch_index
from .mygin import MyGIN
# 【关键】从现在开始，所有 GNN Backbone (比如 GIN) 都必须是 PyG 的版本
# 这是一个示例的 PyG GIN 实现
from torch_geometric.nn import GIN, MLP

# ==========================================================
# 辅助模块 (保持不变，但更清晰)
# ==========================================================
class AttentionAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, Mdim):
        super(AttentionAgger, self).__init__()
        self.model_dim = Mdim
        self.WQ = torch.nn.Linear(Qdim, Mdim)
        self.WK = torch.nn.Linear(Kdim, Mdim)

    def forward(self, Q, K, V, mask=None):
        # Q: [batch_size, Qdim], K/V: [num_subgraphs, Kdim]
        Q, K = self.WQ(Q), self.WK(K) # Q: [B, M], K: [S, M]
        
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim) # [B, S]
        
        if mask is not None:
            # mask 的形状是 [B, S], True 的地方代表需要被掩盖
            Attn = Attn.masked_fill(mask, -1e9) # 用一个很大的负数填充
            
        Attn = torch.softmax(Attn, dim=-1) # [B, S]
        return torch.matmul(Attn, V) # [B, S] * [S, Vdim] -> [B, Vdim]

# ==========================================================
# 主角：Framework (核心改造)
# ==========================================================
class Framework(torch.nn.Module):
    def __init__(self, base_dim, sub_dim, num_class, dropout=0.5):
        super(Framework, self).__init__()
        self.base_model = MyGIN(
              num_node_emb_list=[40], 
              num_edge_emb_list=[10], 
              num_layers=4,           
              emb_dim=base_dim, 
              JK='last', 
              dropout=0.1,            
              readout='mean'
          )
          
        self.sub_model = MyGIN(
            num_node_emb_list=[40], 
            num_edge_emb_list=[10], 
            num_layers=4, 
            emb_dim=sub_dim, 
            JK='last', 
            dropout=0.1, 
            readout='mean'
        )
        
        self.aggr = AttentionAgger(base_dim, sub_dim, max(base_dim, sub_dim))

        self.predictor = MLP([sub_dim, sub_dim, num_class], dropout=dropout, norm=None)

        # 2. 🚨 新增：辅助任务预测头 (Auxiliary Branch)
        # 输入是节点级别的特征 (base_dim)，输出是预测 40 种原子类型
        self.aux_predictor = MLP([base_dim, base_dim, 40], dropout=dropout, norm=None)

    def forward(self, data, return_node_feats=False):
        # --- 1. 处理主图 ---
        # 【修改】：要求 base_model 同时返回节点级特征 (用于辅助任务) 和 图级特征 (用于主任务)
        # 你需要微调一下 MyGIN 的 forward，让它 return node_feats, graph_feats
        node_feats, main_feat = self.base_model(data.x, data.edge_index, data.edge_attr, data.batch) # [num_total_nodes, base_dim]
       
        aux_logits = None
        if return_node_feats:
            # 用辅助头对所有节点进行原子类型预测
            aux_logits = self.aux_predictor(node_feats)
            return aux_logits
        # --- 2. 处理子结构 ---
        # data.subs 是一个 List of Lists
        sub_batch = pyg_batch_from_subgraphs(data.subs) # 拼接成一个大的 Batch 对象
        
        if sub_batch is None: # 如果没有任何子结构
            # 这是一个兜底策略，比如直接用主特征进行预测
            return self.predictor(main_feat)

        sub_batch = sub_batch.to(data.x.device)
        
        _,subs_feat = self.sub_model(
            sub_batch.x, sub_batch.edge_index, sub_batch.edge_attr, sub_batch.batch, 
            
        )
        
        
        
        # --- 3. 注意力聚合 ---
        # 创建一个索引，告诉我们每个子结构属于原始 Batch 里的哪个分子
        sub_to_main_idx = get_subgraph_batch_index(data.subs).to(data.x.device)
        
        # 为了构建注意力 Mask，我们需要知道每个 batch 最大的子结构数量
        num_sub_per_mol = torch.tensor([len(s) for s in data.subs], device=data.x.device)
        max_sub = num_sub_per_mol.max()
        
        # 创建一个 [batch_size, total_subgraphs] 的矩阵，用于将子结构特征映射回分子
        temp_K = torch.zeros(len(data.y), len(subs_feat), device=data.x.device)
        temp_K[sub_to_main_idx, torch.arange(len(subs_feat))] = 1
        
        # K_mapped: [batch_size, max_sub, sub_dim]
        K_mapped = torch.matmul(temp_K, subs_feat).view(len(data.y), -1, subs_feat.size(-1))
        
        # 创建注意力 Mask
        attn_mask = torch.arange(max_sub, device=data.x.device)[None, :] >= num_sub_per_mol[:, None]
        
        # 注意力聚合 (这里我们简化为直接对齐特征)
        # 简化版聚合：对齐特征后，直接相加或拼接
        # 实际代码中，需要一个更复杂的 Attention 模块来处理变长输入
        # 下面是一个更符合原代码逻辑的简化版 Attention
        
        # 将子结构特征按照所属分子聚合
        # scatter_mean 是 PyG 的一个强大工具
        from torch_scatter import scatter_mean
        subs_feat_aggr = scatter_mean(subs_feat, sub_to_main_idx, dim=0) # [batch_size, sub_dim]
        
        # 简单的融合
        molecule_feat = main_feat + subs_feat_aggr # 或者 torch.cat([main_feat, subs_feat_aggr], dim=-1)

        return self.predictor(molecule_feat)


# ==========================================================
# 配角 1：ConditionalGnn (改造)
# ==========================================================
class ConditionalGnn(torch.nn.Module):
    def __init__(self, emb_dim, backend_dim, num_domain, num_class):
        super(ConditionalGnn, self).__init__()
        self.class_emb = torch.nn.Embedding(num_domain, emb_dim)
        self.backend = MyGIN(
            num_node_emb_list=[39], 
            num_edge_emb_list=[10], 
            num_layers=4, 
            emb_dim=backend_dim, 
            JK='last', 
            dropout=0.1, 
            readout='mean'
        )
        self.predictor = MLP([backend_dim + emb_dim, backend_dim, num_class], norm=None)

    def forward(self, data, domains):
        domain_feat = self.class_emb(domains)
        _,graph_feat = self.backend(data.x, data.edge_index, data.edge_attr, data.batch)
      
        
        combined_feat = torch.cat([graph_feat, domain_feat], dim=1)
        return self.predictor(combined_feat)


# ==========================================================
# 配角 2：DomainClassifier (改造)
# ==========================================================
class DomainClassifier(torch.nn.Module):
    def __init__(self, backend_dim, num_domain, num_task=1):
        super(DomainClassifier, self).__init__()
        self.num_task = num_task 
        self.backend = MyGIN(
            num_node_emb_list=[39], 
            num_edge_emb_list=[10], 
            num_layers=4, 
            emb_dim=backend_dim, 
            JK='last', 
            dropout=0.1, 
            readout='mean'
        )
        self.predictor = MLP([backend_dim + num_task, backend_dim, num_domain], norm=None)

    def forward(self, data):
        _,graph_feat = self.backend(data.x, data.edge_index, data.edge_attr, data.batch)
       
        
        y_part = data.y.view(-1, self.num_task).float()
        
        combined_feat = torch.cat([graph_feat, y_part], dim=-1)
        return self.predictor(combined_feat)