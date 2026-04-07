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