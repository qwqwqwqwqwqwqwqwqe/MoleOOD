# 文件路径: models/tta.py (或者 models/utils.py)

import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm # 用于进度条
import torch.nn.functional as F # 用于熵计算可能用到的 F.softmax 等
from models.utils import mask_node_features # 确保导入了节点掩码函数
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

def set_dropout_rate(model: nn.Module, p: float):
    """
    动态修改模型中所有 Dropout 层的丢弃概率。
    这不会改变模型的权重，极其安全！
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p # 强行覆盖原有的 dropout 概率
# ==========================================================
# 1. CoTTA 核心数学工具包
# ==========================================================

def update_ema_variables(ema_model, current_model, alpha_ema=0.99):
    """
    指数滑动平均 (EMA)。导师模型平滑地跟随更新模型。
    """
    for ema_param, current_param in zip(ema_model.parameters(), current_model.parameters()):
        ema_param.data[:] = alpha_ema * ema_param.data[:] + (1.0 - alpha_ema) * current_param.data[:]

@torch.no_grad()
def stochastic_restore(current_model, source_model, p=0.01):
    """
    【对齐论文 Eq. 7 & 8】：随机恢复 (Stochastic Restoration)
    以概率 p 将 current_model 的参数重置为 source_model 的初始权重，防止灾难性遗忘。
    """
    for current_param, source_param in zip(current_model.parameters(), source_model.parameters()):
        if not current_param.requires_grad:
            continue
        # Eq (7): M ~ Bernoulli(p)
        prob_tensor = torch.full_like(current_param, fill_value=p)
        M = torch.bernoulli(prob_tensor).to(current_param.device)
        
        # Eq (8): W_{t+1} = M ⊙ W_0 + (1 - M) ⊙ W_{t+1}
        current_param.data[:] = M * source_param.data[:] + (1.0 - M) * current_param.data[:]

def soft_cross_entropy(logits, target_probs):
    """
    软交叉熵损失。让战士模型去拟合导师模型给出的平滑概率分布。
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()

# ==========================================================
# 2. CoTTA 阈值计算与模型配置
# ==========================================================

@torch.no_grad()
def calculate_source_threshold(model, source_loader, device, delta=0.05):
    """
    【对齐论文 Supplementary】：计算动态阈值 p_th = conf_S - delta。
    conf_S 是源模型在源域数据上预测置信度的 5% 分位数。
    """
    print("\n[CoTTA] 正在源域数据上摸底测验，计算动态增强阈值 p_th ...")
    model.eval()
    all_confs =[]
    
    for data in tqdm(source_loader, desc="Calculating Source Confidences"):
        data = data.to(device)
        logits = model(data)
        probs = torch.softmax(logits, dim=-1)
        confs, _ = torch.max(probs, dim=-1)
        all_confs.append(confs.cpu())
        
    all_confs = torch.cat(all_confs, dim=0)
    conf_S = torch.quantile(all_confs.float(), 0.05).item()
    p_th = conf_S - delta
    
    print(f"[CoTTA] 源域 5% 分位数 conf_S = {conf_S:.4f}, 容差 delta = {delta}")
    print(f"[CoTTA] 💡 最终确定的增强阈值 p_th = {p_th:.4f}\n")
    return p_th

def configure_model_for_cotta(model: nn.Module) -> nn.Module:
    """
    【对齐论文核心】：解锁全参数训练 (Train all trainable parameters)
    因为有 Stochastic Restore 保护，我们可以放心地优化包含分类器在内的所有参数。
    """
    model_for_tta = deepcopy(model)
    model_for_tta.train() # 激活 Dropout (用于数据增强) 和 BN
    
    # 彻底解冻所有参数
    for param in model_for_tta.parameters():
        param.requires_grad = True
        
    # 处理 BN 层：跟踪当前测试流的统计分布
    for m in model_for_tta.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = True
            m.momentum = 0.1 
            
    return model_for_tta

# ==========================================================
# 3. CoTTA 终极主流程
# ==========================================================

@torch.no_grad()
def eval_with_cotta_official(
    model: nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    source_loader: torch.utils.data.DataLoader, # 源域 DataLoader，用于算阈值
    device: torch.device,
    tta_lr: float = 1e-3,          # 作者使用的 1e-3
    alpha_ema: float = 0.99,       # EMA 动量
    restore_prob: float = 0.01,    # 随机恢复的概率 p=0.01
    aug_steps: int = 32,           # 【对齐论文】32 次数据增强
    delta: float = 0.05,           # 【对齐论文】容差 0.05
    evaluate_fn: callable = None
):
    """
    使用完整版 CoTTA (Continual Test-Time Domain Adaptation) 进行评估。
    """
    # 1. 动态计算阈值 p_th
    # p_th = calculate_source_threshold(model, source_loader, device, delta=delta)
    p_th=0.9
    
    # 2. 准备三大模型矩阵
    # A. 源模型 W_0 (定海神针)：用于随机恢复，绝对冻结
    source_model = deepcopy(model).to(device)
    source_model.eval()
    for p in source_model.parameters(): p.requires_grad = False

    # B. 导师模型 W_ema：用于生成软伪标签，冻结
    ema_model = deepcopy(model).to(device)
    for p in ema_model.parameters(): p.requires_grad = False

    # C. 战士模型 W_t (更新模型)：解冻所有参数，准备微调
    current_model = configure_model_for_cotta(model).to(device)
    
    # 使用 Adam 优化器
    optimizer = torch.optim.Adam(current_model.parameters(), lr=tta_lr)
    
    result_all, gt_all = [],[]
    
    for data in tqdm(test_loader, desc="CoTTA Continual Adaptation"):
        data = data.to(device)
        
        # ---------------------------------------------------------
        # 阶段一：导师模型生成高质量软伪标签 (Augmentation-Averaged)
        # ---------------------------------------------------------
        with torch.no_grad():
            ema_model.eval() # 🚨 保持冷静！全程关闭破坏性的标准 Dropout！
            
            # 1. 基础预测 (不带任何扰动)
            standard_logits = ema_model(data, aug_type=None)
            standard_probs = torch.softmax(standard_logits, dim=-1)
            standard_confs, _ = torch.max(standard_probs, dim=-1)
            
            need_aug_mask = standard_confs <= p_th
            final_ema_probs = standard_probs.clone()
            
            # 2. 如果存在迷茫样本，启动高级特征空间增强！
            if need_aug_mask.sum() > 0:
                aug_probs_sum = 0
                
                # 设定你想要的增强方式和强度
                # 推荐：'noise' (aug_ratio=0.05~0.1) 或者 'mask' (aug_ratio=0.1~0.2)
                selected_aug_type = 'noise' 
                selected_aug_ratio = 0.05  
                
                for _ in range(aug_steps): # 跑 32 次
                    # 每次前向传播，底层 Embedding 都会被随机加上不同的高斯噪声
                    aug_logits = ema_model(data, aug_type=selected_aug_type, aug_ratio=selected_aug_ratio)
                    aug_probs_sum += torch.softmax(aug_logits, dim=-1)
                
                aug_probs_mean = aug_probs_sum / aug_steps
                final_ema_probs[need_aug_mask] = aug_probs_mean[need_aug_mask]
                
            final_ema_probs = final_ema_probs.detach()

        # ---------------------------------------------------------
        # 阶段二：战士模型更新参数 (Self-Training)
        # ---------------------------------------------------------
        current_model.train()
        with torch.enable_grad():
            logits = current_model(data)
            loss = soft_cross_entropy(logits, final_ema_probs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 得到初步的 W_{t+1}
            
        # ---------------------------------------------------------
        # 阶段三：CoTTA 状态维护 (EMA & Stochastic Restore)
        # ---------------------------------------------------------
        # 1. 导师跟随战士
        update_ema_variables(ema_model, current_model, alpha_ema)
        # 2. 战士回归本源 (公式 7 & 8，解决灾难性遗忘)
        stochastic_restore(current_model, source_model, p=restore_prob)

        # ---------------------------------------------------------
        # 阶段四：最终推理作答
        # ---------------------------------------------------------
        current_model.eval() # 答题时必须关闭 Dropout
        with torch.no_grad():
            final_result = torch.softmax(current_model(data), dim=-1)
            
        result_all.append(final_result.detach().cpu())
        gt_all.append(data.y.cpu().view(-1))
        
    # 汇总输出
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    
    if evaluate_fn:
        return evaluate_fn(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])
    return result_all, gt_all

@torch.no_grad()
def eval_with_masked_cotta(
    model: nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device,
    tta_lr: float = 1e-4,          # 调小学习率，1e-3 对 GNN 太暴躁了
    alpha_ema: float = 0.99,
    restore_prob: float = 0.01,
    aug_steps: int = 32,
    p_th: float = 0.8,             # 【修改】直接使用硬性高阈值 0.8
    evaluate_fn: callable = None
):
    print(f"\n[TTA] 启动 Masked CoTTA (二分类适配版: p_th={p_th}, lr={tta_lr}, augs={aug_steps})...")
    
    # ... (模型拷贝和配置与之前相同) ...
    source_model = deepcopy(model).to(device)
    source_model.eval()
    for p in source_model.parameters(): p.requires_grad = False

    ema_model = deepcopy(model).to(device)
    for p in ema_model.parameters(): p.requires_grad = False

    current_model = configure_model_for_cotta(model).to(device)
    optimizer = torch.optim.Adam(current_model.parameters(), lr=tta_lr)
    
    result_all, gt_all = [],[]
    
    for data in tqdm(test_loader, desc="Masked CoTTA"):
        data = data.to(device)
        
        # ---------------------------------------------------------
        # 阶段一：导师模型生成伪标签 (MC Dropout + 强阈值)
        # ---------------------------------------------------------
        with torch.no_grad():
            ema_model.train() # 强制开启 Dropout，始终使用 32 次增强
            aug_probs_sum = 0
            for _ in range(aug_steps):
                aug_probs_sum += torch.softmax(ema_model(data), dim=-1)
            
            # 获取经过 32 次深思熟虑后的平均概率
            ema_probs = aug_probs_sum / aug_steps
            
            # 计算最高置信度
            max_confs, _ = torch.max(ema_probs, dim=-1)
            
            # 【核心修改】：生成布尔掩码，只挑选极其确定的样本！
            valid_mask = max_confs > p_th

        # ---------------------------------------------------------
        # 阶段二：战士模型更新 (过滤毒数据)
        # ---------------------------------------------------------
        current_model.train()
        
        # 【核心修改】：如果这个 Batch 里连一个大于 0.8 的自信样本都没有，
        # 我们就直接跳过梯度更新！宁可不学，绝不学错！
        if valid_mask.sum() > 0:
            with torch.enable_grad():
                logits = current_model(data)
                
                # 只拿高质量的样本算 Loss
                valid_logits = logits[valid_mask]
                valid_ema_probs = ema_probs[valid_mask].detach()
                
                loss = soft_cross_entropy(valid_logits, valid_ema_probs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        # ---------------------------------------------------------
        # 阶段三 & 四：EMA, Restore 与 预测 (保持不变)
        # ---------------------------------------------------------
        update_ema_variables(ema_model, current_model, alpha_ema)
        stochastic_restore(current_model, source_model, p=restore_prob)

        current_model.eval()
        with torch.no_grad():
            final_result = torch.softmax(current_model(data), dim=-1)
            
        result_all.append(final_result.detach().cpu())
        gt_all.append(data.y.cpu().view(-1))

    # ... (汇总返回结果) ...
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    if evaluate_fn:
        return evaluate_fn(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])
    return result_all, gt_all

