import os
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
import torch.nn as nn
import random
# 导入你的核心组件 (请确保路径正确)
from torch_geometric.loader import DataLoader
from models.dataset import LBAPDatasetWithSub
from models.Framework import Framework
import numpy as np
from models.utils import evaluate, mask_node_features
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果用多卡
    # 🚨 极其关键：强制让 GPU 算法变得确定（虽然会慢一点）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def init_args():
    parser = argparse.ArgumentParser('Drug-TTA (Meta-Auxiliary) Evaluation Script')
    
    # --- 核心路径参数 ---
    parser.add_argument('--model_path', type=str, required=True, help="训练好的 .pth 文件路径")
    parser.add_argument('--test_path', type=str, required=True, help="测试集 JSON 文件路径")
    
    # --- 模型架构参数 (必须与训练时完全一致！) ---
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--dropout', default=0.1, type=float) 
    parser.add_argument('--framework_dropout', default=0.5, type=float) 

    # --- 数据与硬件参数 ---
    parser.add_argument('--batch_size', default=128, type=int) # 推理时可以适度调大
    parser.add_argument('--device', default=0, type=int)
    
    # --- TTA 专属超参数 (在这里尽情调参吧！) ---
    parser.add_argument('--tta_lr', default=1e-3, type=float, help="TTA 阶段辅助任务的学习率")
    parser.add_argument('--tta_steps', default=1, type=int, help="每个 Batch 适应的步数")
    parser.add_argument('--mask_rate', default=0.15, type=float, help="辅助任务的节点掩码比例")
    
    parser.add_argument('--seed', default=2022, type=int, help="随机种子，确保实验可重复")
    return parser.parse_args()

@torch.no_grad()
def eval_with_meta_auxiliary(
    model: torch.nn.Module, 
    loader: DataLoader, 
    device: torch.device,
    tta_lr: float,
    tta_steps: int,
    mask_rate: float
):
    """
    完全对齐 Drug-TTA 论文的 Testing Stage:
    对每个测试 Batch，先用辅助任务更新 BN 层，再执行主任务预测。
    """
    print(f"\n[TTA] 🚀 启动 Meta-Auxiliary TTA (steps={tta_steps}, lr={tta_lr}, mask_rate={mask_rate})...")
    
    # 1. 准备专门用于 TTA 的模型副本
    model_for_tta = deepcopy(model).to(device)
    
    # 2. 严格遵循论文逻辑：冻结所有参数，只解冻 BN 层和辅助头
    for param in model_for_tta.parameters():
        param.requires_grad = False
        
    bn_and_aux_params = []
    for name, m in model_for_tta.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            # m.momentum = 0.01
            for param in m.parameters():
                bn_and_aux_params.append(param)
        elif 'aux_predictor' in name:
            for param in m.parameters():
                param.requires_grad = True
                bn_and_aux_params.append(param)

    result_all, gt_all = [], []
    
    for data in tqdm(loader, desc="Meta-Aux TTA Testing"):
        data = data.to(device)
        
        # 🚨 Episodic 模式：每次面对新 Batch，恢复到训练结束时的最佳状态！
        # 这是 TTA 不崩溃的核心保障
        model_for_tta.load_state_dict(model.state_dict())
        
        # 每次重置优化器状态，清除历史动量
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_for_tta.parameters()), 
            lr=tta_lr
        )
        
        # =====================================================================
        # 🚀 Step 1: 测试时内循环 (Updating Encoder Layer Norm with Auxiliary Loss)
        # =====================================================================
        model_for_tta.train() # 必须开启 train()，BN 层才能计算当前 Batch 的统计量
        
        for _ in range(tta_steps):
            # 制造辅助任务输入 (遮挡节点)
            masked_data, mask_idx, masked_labels = mask_node_features(data, mask_rate=mask_rate)
            
            if mask_idx is not None and len(mask_idx) > 0:
                with torch.enable_grad(): # 临时开启梯度计算
                    # 仅执行辅助预测
                    aux_logits = model_for_tta(masked_data, return_node_feats=True)
                    aux_loss = F.cross_entropy(aux_logits[mask_idx], masked_labels)
                    
                    optimizer.zero_grad()
                    aux_loss.backward()
                    optimizer.step() # BN 层和辅助头被当前测试分布完美预热！

        # =====================================================================
        # 🚀 Step 2: 测试时主任务预测 (Inference with Updated Encoder)
        # =====================================================================
        model_for_tta.eval() # 预热完毕！切换到评估模式，关闭 Dropout
        with torch.no_grad():
            # 使用预热好的 BN 层和原始图数据进行毒性预测
            main_logits = model_for_tta(data, return_node_feats=False)
            main_probs = torch.softmax(main_logits, dim=-1)
            
        result_all.append(main_probs.detach().cpu())
        gt_all.append(data.y.cpu().view(-1))

    # --- 汇总结果 ---
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    
    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])

@torch.no_grad()
def eval_with_zero_shot_meta_aux(
    model: torch.nn.Module, 
    loader: torch.utils.data.DataLoader, 
    device: torch.device,
    
    tta_lr: float = 0.01,   # 沿用你跑出大提升的猛药
    tta_steps: int = 3,     # 沿用你的成功经验
    mask_rate: float = 0.4
):
    print(f"\n[TTA] 🚀 启动零样本外挂 Meta-Aux TTA (steps={tta_steps}, lr={tta_lr}, mask={mask_rate})...")
    
    model_for_tta = deepcopy(model).to(device)
    
    # =================================================================
    # 🚨 终极魔法：临时强行外挂一个辅助预测头！
    # 因为原始的 best_model.pth 里没有这个头，我们在这里当场造一个。
    # 假设 base_dim 是 128，原子种类是 40
    # =================================================================
    base_dim=128
    # 我们用一个极简的单层 Linear，防止它过度拟合，只提供最基础的梯度方向
    aux_head = nn.Linear(base_dim, 40).to(device) 
    
    # 冻结主干，只解冻 BN 层和这颗新装上的临时“眼睛”
    for param in model_for_tta.parameters():
        param.requires_grad = False
        
    bn_params = []
    for m in model_for_tta.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True
            m.momentum = 0.1
            for param in m.parameters():
                bn_params.append(param)
                
    # 把 BN 层参数和外挂辅助头的参数一起交给优化器
    # 这样，辅助头在努力看清被遮挡的原子的同时，它的梯度就会流过 BN 层，完美预热它！
    optimizer = torch.optim.Adam(bn_params + list(aux_head.parameters()), lr=tta_lr)

    result_all, gt_all = [], []
    
    for data in tqdm(loader, desc="Zero-Shot Meta-Aux TTA"):
        data = data.to(device)
        
        # 🚨 Episodic 模式：防止在这极端的 0.01 学习率下模型崩溃
        model_for_tta.load_state_dict(model.state_dict())
        
        # 每次面临新 Batch，重置优化器，同时重置我们外挂的辅助头！
        # 确保每个 Batch 都在同一个起跑线上预热
        aux_head = nn.Linear(base_dim, 40).to(device)
        optimizer = torch.optim.Adam(bn_params + list(aux_head.parameters()), lr=tta_lr)
        
        model_for_tta.train() 
        
        for _ in range(tta_steps):
            masked_data, mask_idx, masked_labels = mask_node_features(data, mask_rate=mask_rate)
            
            if mask_idx is not None and len(mask_idx) > 0:
                with torch.enable_grad():
                    # 1. 提取被遮挡数据的节点特征
                    # 注意：你需要确保 base_model 返回了 node_feats
                    node_feats, _ = model_for_tta.base_model(
                        masked_data.x, masked_data.edge_index, 
                        masked_data.edge_attr, masked_data.batch
                    )
                    
                    # 2. 用外挂头进行预测
                    aux_logits = aux_head(node_feats)
                    aux_loss = F.cross_entropy(aux_logits[mask_idx], masked_labels)
                    
                    # 3. 反向传播，更新外挂头和 BN 层！
                    optimizer.zero_grad()
                    aux_loss.backward()
                    optimizer.step()

        # --- TTA 结束，进入严肃答题模式 ---
        model_for_tta.eval()
        with torch.no_grad():
            main_logits = model_for_tta(data)
            main_probs = torch.softmax(main_logits, dim=-1)
            
        result_all.append(main_probs.detach().cpu())
        gt_all.append(data.y.cpu().view(-1))
    # --- 汇总结果 ---
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    
    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])
if __name__ == '__main__':
    
    args = init_args()
    seed_everything(args.seed)
    print("--- 启动 Drug-TTA 独立评估脚本 ---")
    print(args)

    device = torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}')

    # 1. 构建空壳模型
    print("\n[INFO] Building model architecture...")
    main_model = Framework(
        base_dim=args.emb_dim, 
        sub_dim=args.emb_dim, 
        num_class=args.num_class, 
        dropout=args.framework_dropout
    ).to(device)

     # =====================================================================
    # 2. 灵魂附体与完美缝合 (加载 0.6691 的旧权重)
    # =====================================================================
    print(f"\n[INFO] Loading pre-trained weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    checkpoint = torch.load(args.model_path, map_location=device)
    pretrained_dict = checkpoint['main']
    model_dict = main_model.state_dict()
    
    # --- A. 过滤掉新模型里有，但旧模型里没有的层 (比如 aux_predictor) ---
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    # --- B. 缝合尺寸不匹配的层 (处理 39 -> 40 的 Embedding 扩充) ---
    import math # 确保导入了 math
    for k, v in filtered_dict.items():
        if v.shape != model_dict[k].shape:
            print(f"[Surgery] 正在缝合尺寸不匹配的层: {k} (旧: {v.shape} -> 新: {model_dict[k].shape})")
            # 处理 nn.Linear(39, 128) 扩展为 nn.Linear(40, 128) 的权重矩阵
            # 权重矩阵形状为 [out_features, in_features]
            if len(v.shape) == 2 and v.shape[0] == model_dict[k].shape[0] and v.shape[1] < model_dict[k].shape[1]:
                # 创建新壳，形状对齐新模型 [128, 40]
                new_w = torch.empty_like(model_dict[k])
                # Kaiming 随机初始化整个新壳 (给第 40 列合理的初始值)
                torch.nn.init.kaiming_uniform_(new_w, a=math.sqrt(5))
                # 极其关键：将旧模型的 39 列完美覆盖过去
                new_w[:, :v.shape[1]] = v
                filtered_dict[k] = new_w
                print(f"  -> [成功] 旧权重已保留，新增的 MASK 维度已随机初始化。")
            else:
                print(f"  -> [警告] 无法自动缝合，保持新模型的随机初始化。")
                filtered_dict[k] = model_dict[k]

    # --- C. 执行最终的加载 ---
    model_dict.update(filtered_dict)
    # strict=False 允许 model_dict 中有没被更新的 key (如 aux_predictor)
    main_model.load_state_dict(model_dict, strict=False)
    print("✅ 权重加载与缝合成功！")


    # 3. 加载测试数据
    print(f"\n[INFO] Loading test data from: {args.test_path}")
    test_set  = LBAPDatasetWithSub(args.test_path, split='ood_test')
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 4. 执行 Meta-Auxiliary TTA 评估
    test_perf = eval_with_zero_shot_meta_aux(
        model=main_model, 
        loader=test_loader, 
        device=device,
        tta_lr=args.tta_lr,
        tta_steps=args.tta_steps,
        mask_rate=args.mask_rate
    )

    # 5. 打印最终成绩
    print("\n" + "="*50)
    print("🎉 最终评估结果 (FINAL EVALUATION RESULT) 🎉")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Test Dataset: {args.test_path}")
    print(f"TTA LR: {args.tta_lr} | TTA Steps: {args.tta_steps} | Mask Rate: {args.mask_rate}")
    print("-" * 50)
    print(f"🔥 AUC:      {test_perf['auc']:.4f}")
    print(f"🎯 Accuracy: {test_perf['accuracy']:.4f}")
    print("="*50)