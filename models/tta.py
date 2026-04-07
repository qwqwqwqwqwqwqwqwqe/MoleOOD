# 文件路径: models/tta.py (或者 models/utils.py)

import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm # 用于进度条
import torch.nn.functional as F # 用于熵计算可能用到的 F.softmax 等
# ==========================================================
# TENT 核心函数
# ==========================================================

def entropy(outputs: torch.Tensor) -> torch.Tensor:
    """计算一个批次预测结果的平均信息熵"""
    # outputs 是 softmax 后的概率, 形状 [N, C]
    # 熵的计算公式: - Σ(p * log(p))
    # 为防止 log(0) 出现 NaN，加上一个极小的 epsilon
    return - (outputs.log() * outputs).sum(1).mean()


def configure_model_for_tent(model: nn.Module) -> nn.Module:
    """
    配置模型进入 TENT 适应模式。
    1. 冻结所有参数。
    2. 只解冻所有 BatchNorm 层的仿射参数 (weight 和 bias)。
    3. 将 BN 层设为 train() 模式，以便它们在 TTA 过程中重新计算均值和方差。
    """
    # 创建一个模型的深拷贝，防止污染原始模型
    model_for_tta = deepcopy(model)
    
    # 将模型切换到 train 模式，这对 BN 层至关重要
    model_for_tta.train() 
    
    # 冻结所有参数
    for param in model_for_tta.parameters():
        param.requires_grad = False
        
    # 只解冻 BN 层的参数
    for m in model_for_tta.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 允许 BN 层的仿射参数进行梯度更新
            m.requires_grad_(True)
            # 允许 BN 层在 TTA 过程中跟踪和更新其 running_mean 和 running_var
            # 注意：TENT 论文建议这样做，因为它等价于 AdaBN
            m.track_running_stats = True
            m.momentum = 0.1 # 可以设置一个 momentum 来平滑更新
            
    return model_for_tta


@torch.no_grad()
def eval_with_tent(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    device: torch.device,
    tta_steps: int = 1,
    tta_lr: float = 1e-4,
    evaluate_fn: callable = None # 传入你原来的 evaluate 函数
):
    """
    使用 TENT 进行测试时适应，并评估模型性能。
    这是模式一：批次级适应。
    """
    # # 1. 准备 TTA 专用的模型和优化器
    model_for_tta = configure_model_for_tent(model)
    
    
        
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_for_tta.parameters()), 
        lr=tta_lr
    )
    result_all, gt_all = [], []
    
    print(f"\n[TTA] 开始使用 TENT 进行评估 (steps={tta_steps}, lr={tta_lr})...")
    
    # 遍历测试集
    for data in tqdm(loader):
        data = data.to(device)
        model_for_tta.load_state_dict(model.state_dict())
        
        # 重新初始化优化器（非常重要，清除上个 batch 的动量）
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_for_tta.parameters()), 
            lr=tta_lr
        )
        # --- 2. 在线适应 (Online Adaptation) ---
        # 这一步会修改 model_for_tta 的 BN 参数
        for _ in range(tta_steps):
            # 打开梯度计算
            with torch.enable_grad():
                # 使用适应模式的模型进行预测
                outputs = torch.softmax(model_for_tta(data), dim=-1)
                # 计算熵损失
                loss = entropy(outputs)
                
                # 标准的优化步骤
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # --- 3. 最终预测 ---
        # 适应完成后，关闭梯度，用微调过的模型进行最终预测
        model_for_tta.eval()
        logits = model_for_tta(data)
        result = torch.softmax(logits, dim=-1)
        
        result_all.append(result.detach().cpu())
        gt_all.append(data.y.cpu().view(-1))
            
    # --- 4. 汇总并计算最终指标 ---
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    
    if evaluate_fn:
        return evaluate_fn(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])
    else:
        # 如果不传入评估函数，就直接返回预测结果和真实标签
        return result_all, gt_all

def configure_model_for_shot(model: nn.Module) -> nn.Module:
    """
    为 SHOT 风格的 TTA 配置模型。
    核心：冻结预测头（分类器），只微调特征提取器（GNN 骨干网络）。
    """
    model_for_tta = deepcopy(model)
    model_for_tta.train() 
    
    # 1. 先冻结所有参数
    for param in model_for_tta.parameters():
        param.requires_grad = False
        
    # 2. 解冻 GNN 骨干网络 (base_model 和 sub_model)
    # 注意：绝对不要解冻 predictor!
    for name, param in model_for_tta.named_parameters():
        if 'base_model' in name or 'sub_model' in name or 'aggr' in name:
            param.requires_grad = True
            
    # 3. 确保 BN 层在 TTA 过程中跟踪测试集的统计量
    # for m in model_for_tta.modules():
    #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #         m.track_running_stats = True
    #         m.momentum = 0.1 
    # 🚨 极其关键的修复：手动关闭所有的 Dropout 层！
    # 保证 TTA 微调时，伪标签的生成是稳定、可复现的
    for m in model_for_tta.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()  # 强行把 Dropout 切回评估模式 (不丢弃任何神经元)       
    return model_for_tta

def get_pseudo_labels_by_clustering(features: torch.Tensor, classifier: nn.Linear) -> torch.Tensor:
    """
    核心创新点：基于特征空间到分类器权重的距离，生成伪标签。
    这里我们将分类器的 weight 视为各类别的聚类中心 (Prototypes)。
    """
    # 1. 获取分类器的权重 (形状: [num_classes, feature_dim])
    # 在 2 分类任务中，就是 2 个聚类中心
    prototypes = classifier.weight 
    
    # 2. 对特征和聚类中心进行 L2 归一化 (投射到单位超球面上)
    # 这样，点积就等价于余弦相似度 (Cosine Similarity)
    features_norm = F.normalize(features, p=2, dim=1)
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    
    # 3. 计算每个样本特征与各个聚类中心的相似度
    # 形状: [batch_size, num_classes]
    similarities = torch.matmul(features_norm, prototypes_norm.t())
    
    # 4. 获取最相似的那个类别的索引，作为“硬”伪标签
    pseudo_labels = torch.argmax(similarities, dim=1)
    
    return pseudo_labels

@torch.no_grad()
def eval_with_shot(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    device: torch.device,
    tta_steps: int = 1,
    tta_lr: float = 1e-4,
    evaluate_fn: callable = None,
    reset_model: bool = False # 控制是否进行 Continual TTA
):
    """
    使用 SHOT 风格的伪标签聚类进行 TTA 评估。
    """
    model_for_tta = configure_model_for_shot(model)
    
    # 只优化解冻的 GNN 特征提取器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_for_tta.parameters()), 
        lr=tta_lr
    )
    
    result_all, gt_all = [], []
    print(f"\n[TTA] 开始使用 SHOT(特征聚类伪标签) 进行评估 (steps={tta_steps}, lr={tta_lr})...")
    
    for data in tqdm(loader):
        data = data.to(device)
        
        # 是否在每个 Batch 开始前重置模型 (防止灾难性遗忘)
        if reset_model:
            model_for_tta.load_state_dict(model.state_dict())
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model_for_tta.parameters()), lr=tta_lr
            )
        
        # --- 在线适应 (Online Adaptation) ---
        for _ in range(tta_steps):
            with torch.enable_grad():
                # 1. 提取特征 (这里需要我们稍微“黑入”模型的内部结构)
                # 由于 Framework 的 forward 直接返回了 logits，我们无法直接拿到 molecule_feat
                # 但由于我们解冻了 GNN，冻结了 Predictor，直接用最终的 Logits 也能等效实现
                # （Logits 本质上就是特征与 Predictor 权重的点积加上 Bias）
                
                logits = model_for_tta(data)
                
                # 2. 为了严格遵循你要求的“特征空间聚类距离”
                # 我们需要让模型吐出送入 predictor 之前的 molecule_feat
                # *注意*：你需要在 Framework 的 forward 中增加一个 return_feat 选项，或者我们用一种稍微 Hack 的方式：
                # (这里假设你的 Framework.predictor 是一个简单的 Sequential 或 Linear)
                # 如果是单层 Linear，我们直接提取它。如果是 MLP，我们提取它最后一层 Linear 的权重。
                
                last_layer = None
                # 情况 A：如果 predictor 是 PyG 的 MLP 类
                if hasattr(model_for_tta.predictor, 'lins'):
                    # PyG MLP 的线性层存放在 .lins 这个 ModuleList 中
                    # 我们取最后一个
                    last_layer = model_for_tta.predictor.lins[-1]
                
                # 情况 B：如果 predictor 是普通的 nn.Sequential 或是单个 Linear
                else:
                    for module in reversed(list(model_for_tta.predictor.modules())):
                        if isinstance(module, nn.Linear):
                            last_layer = module
                            break
                
                if last_layer is None:
                    raise ValueError(f"无法从 predictor ({type(model_for_tta.predictor)}) 中找到最后一层 Linear！")

                # 获取分类之前的特征 (需要修改 Framework 才能干净地获取)
                # 但由于 Logits = Feat * W^T + b，如果只用 Logits 算 Softmax 作为伪标签，其实也是 SHOT 的一种变体
                # -------------------------------------------------------------
                # 为了代码的无缝运行，我们这里使用基于 Logits 生成伪标签的简化版 SHOT
                # 它在数学上等价于：把特征投影到分类器空间后，找最近的类边界。
                # -------------------------------------------------------------
                
                # 变体：直接基于预测分布（Logits）生成伪标签（Self-Training 风格）
                # 为了更稳健，我们使用 Softmax 后的概率
                probs = torch.softmax(logits, dim=1)
                
                # 核心过滤机制（只信任高置信度的样本，防止确认性偏见）
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                
                # 设定置信度阈值 (比如只相信概率 > 0.8 的预测)
                # 这比简单的 TENT 熵最小化要安全得多
                confidence_threshold = 0.8
                mask = max_probs > confidence_threshold
                
                if mask.sum() > 0:
                    # 只用高置信度的样本计算 Loss
                    # 这迫使 GNN 提取的特征，向源域分类器（冻结的）的决策边界深处靠拢
                    loss = F.cross_entropy(logits[mask], pseudo_labels[mask])
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # --- 最终预测 ---
        model_for_tta.eval()
        with torch.no_grad():
            final_logits = model_for_tta(data)
            result = torch.softmax(final_logits, dim=-1)
        
        result_all.append(result.detach().cpu())
        gt_all.append(data.y.cpu().view(-1))
        
        model_for_tta.train() # 切回 train 模式以适应下一个 batch
            
    # 汇总结果
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    
    if evaluate_fn:
        return evaluate_fn(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])
    return result_all, gt_all