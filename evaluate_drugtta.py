import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from copy import deepcopy
import math

from torch_geometric.loader import DataLoader
from models.dataset import LBAPDatasetWithSub
from models.Framework import Framework
from models.utils import evaluate, mask_node_features, pyg_batch_from_subgraphs

def init_args():
    parser = argparse.ArgumentParser('Drug-TTA (Zero-Shot Meta-Auxiliary) Evaluation Script')
    
    # 路径参数
    parser.add_argument('--model_path', type=str, required=True, help="纯 MoleOOD 的 best_model.pth 路径")
    parser.add_argument('--train_path', type=str, required=True, help="用于训练辅助头的源域数据 (如 train set JSON)")
    parser.add_argument('--test_path', type=str, required=True, help="测试集 JSON 文件路径")
    
    # 模型架构参数 (必须与 0.6691 模型完全一致)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--dropout', default=0.1, type=float) 
    parser.add_argument('--framework_dropout', default=0.5, type=float) 

    # 数据与硬件
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    
    # --- TTA 外挂专属超参数 ---
    parser.add_argument('--adapt_epochs', default=5, type=int, help="在源域上训练辅助头的轮数 (不需要太多)")
    parser.add_argument('--adapt_lr', default=1e-3, type=float, help="训练辅助头的学习率")
    # parser.add_argument('--tta_lr', default=0.01, type=float, help="TTA 阶段测试时预热 BN 层的学习率")
    # parser.add_argument('--tta_steps', default=3, type=int, help="每个测试 Batch 预热的步数")
    parser.add_argument('--mask_rate', default=0.2, type=float, help="节点掩码比例 (你跑出的黄金 0.2)")
    # parser.add_argument('--tta_lr', default=1e-3, type=float, help="主模型更新学习率（main loss）")
    parser.add_argument('--aux_lr', default=1e-3, type=float, help="aux更新学习率")
    parser.add_argument('--inner_steps', default=3, type=int, help="inner adaptation 步数")
    parser.add_argument('--test_aux_lr', default=1e-3, type=float, help="测试时aux更新学习率")
    parser.add_argument('--bn_lr', default=1e-3, type=float, help="bn更新学习率")
    return parser.parse_args()

def get_bn_aux_params(model):
    params = []
    seen = set()

    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for p in m.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        elif 'aux_predictor' in name:
            for p in m.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

    return params
def get_aux_bn_groups(model):
    aux_params = []
    bn_params = []

    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_params += list(m.parameters())
        elif 'aux_predictor' in name:
            aux_params += list(m.parameters())

    return aux_params, bn_params
def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================================
# 核心阶段一：在源域上训练“专属外挂” (TTA Adaptation Phase)
# =====================================================================
def train_aux_head_only(model, loader, device, lr, epochs, mask_rate):
    """
    极简、无损的辅助头训练。
    彻底冻结预训练的 GNN 和主分类头，只训练 aux_predictor 和 BN 层。
    """
    print(f"\n[🚀 ADAPTATION] 开始在源域上训练专属辅助探针 (Epochs={epochs}, LR={lr})...")
    model.train()
    
    # 1. 严格分离参数
    bn_and_aux_params = []
    for name, param in model.named_parameters():
        if 'bn' in name or 'batch_norm' in name or 'aux_predictor' in name:
            param.requires_grad = True
            bn_and_aux_params.append(param)
        else:
            param.requires_grad = False # 死死锁住 0.6691 的灵魂！

    optimizer = torch.optim.Adam(bn_and_aux_params, lr=lr)
    
    for ep in range(epochs):
        epoch_loss = 0
        for data in tqdm(loader, desc=f"Adapt Epoch {ep+1}/{epochs}"):
            data = data.to(device)
            masked_data, mask_idx, masked_labels = mask_node_features(data, mask_rate=mask_rate)
            
            if mask_idx is not None and len(mask_idx) > 0:
                
                
                # 获取被遮挡节点的预测特征
                aux_logits = model(masked_data, return_node_feats=True)
                
                
                loss = F.cross_entropy(aux_logits[mask_idx], masked_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        print(f"   -> Epoch {ep+1} Aux Loss: {epoch_loss/len(loader):.4f}")
    
    print("✅ 专属辅助探针训练完毕！模型已具备极其敏锐的分布感知能力。")
    return model

# =====================================================================
# 核心阶段二：带兵出征测试集 (TTA Inference Phase)
# =====================================================================

def eval_with_meta_auxiliary(model, loader, device, bn_lr, aux_lr, test_aux_lr, inner_steps, mask_rate):
    print(f"\n[TTA] 🚀 Meta-Aux TTA (steps={inner_steps}, lr={aux_lr}, mask_rate={mask_rate})...")
    
    model_for_tta = deepcopy(model).to(device)

    # 只收集 BN + aux 参数
    bn_and_aux_params = []
    for name, m in model_for_tta.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for param in m.parameters():
                param.requires_grad = True
                bn_and_aux_params.append(param)
        elif 'aux_predictor' in name:
            for param in m.parameters():
                param.requires_grad = True
                bn_and_aux_params.append(param)

    result_all, gt_all = [], []

    for data in tqdm(loader, desc="Meta-Aux TTA Testing"):
        data = data.to(device)

        # 每个 batch reset
        model_for_tta.load_state_dict(model.state_dict())

        optimizer_all = torch.optim.Adam(model_for_tta.parameters(), lr=bn_lr)
        aux_params, bn_params = get_aux_bn_groups(model_for_tta)
        optimizer_bn = torch.optim.Adam(get_aux_bn_groups(model)[1], lr=bn_lr)
        optimizer_aux = torch.optim.SGD([
            {'params': aux_params, 'lr': args.aux_lr},
            {'params': bn_params, 'lr': args.bn_lr}
        ], momentum=0.9)

        # =====================================================
        # 🔁 阶段1：inner_steps 训练
        # =====================================================
        model_for_tta.train()

        for _ in range(inner_steps):

            # ---- Step 1: aux loss ----
            masked_data, mask_idx, masked_labels = mask_node_features(data, mask_rate=mask_rate)
            if mask_idx is not None and len(mask_idx) > 0:
                aux_logits = model_for_tta(masked_data, return_node_feats=True)
                aux_loss = F.cross_entropy(aux_logits[mask_idx], masked_labels)

                optimizer_aux.zero_grad()
                aux_loss.backward()
                optimizer_aux.step()

            # # ---- Step 2: main loss ----
            pred = model_for_tta(data, return_node_feats=False)
            # main_loss = F.cross_entropy(pred, data.y.view(-1))
            probs = torch.softmax(pred, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            main_loss = entropy

            # optimizer_all.zero_grad()
            # main_loss.backward()
            # optimizer_all.step()
            optimizer_bn.zero_grad()
            entropy.backward()
            optimizer_bn.step()

        # =====================================================
        # 🎯 阶段2：测试（只一轮）
        # =====================================================

        # ❗先 aux 更新一次（但只更新 aux + BN）
        model_for_tta.train()
        optimizer_aux = torch.optim.Adam(
            get_bn_aux_params(model_for_tta), lr=test_aux_lr
        )
        masked_data, mask_idx, masked_labels = mask_node_features(data, mask_rate=mask_rate)
        if mask_idx is not None and len(mask_idx) > 0:
            aux_logits = model_for_tta(masked_data, return_node_feats=True)
            aux_loss = F.cross_entropy(aux_logits[mask_idx], masked_labels)

            optimizer_aux.zero_grad()
            aux_loss.backward()
            optimizer_aux.step()

        # ❗推理：关闭 dropout，但 BN 用当前 batch
        model_for_tta.eval()
        for m in model_for_tta.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
                m.track_running_stats = True
                m.momentum = 0.1
        with torch.no_grad():
            main_logits = model_for_tta(data, return_node_feats=False)
            main_probs = torch.softmax(main_logits, dim=-1)

        result_all.append(main_probs.cpu())
        gt_all.append(data.y.cpu().view(-1))

    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)

    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])

# =====================================================================
# 流程编排总导演
# =====================================================================
if __name__ == '__main__':
    args = init_args()
    seed_everything(args.seed)
    device = torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}')

    # 1. 建空壳 (带辅助头，且字典扩展到 40 维)
    main_model = Framework(base_dim=args.emb_dim, sub_dim=args.emb_dim, num_class=args.num_class, dropout=args.framework_dropout).to(device)

    # 2. 完美缝合 (加载 0.6691)
    print(f"\n[INFO] 正在缝合预训练权重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    pretrained_dict = checkpoint['main']
    model_dict = main_model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    for k, v in filtered_dict.items():
        if v.shape != model_dict[k].shape:
            if len(v.shape) == 2 and v.shape[0] == model_dict[k].shape[0] and v.shape[1] < model_dict[k].shape[1]:
                new_w = torch.empty_like(model_dict[k])
                torch.nn.init.kaiming_uniform_(new_w, a=math.sqrt(5))
                new_w[:, :v.shape[1]] = v
                filtered_dict[k] = new_w
            else:
                filtered_dict[k] = model_dict[k]

    model_dict.update(filtered_dict)
    main_model.load_state_dict(model_dict, strict=False)
    print("✅ 权重完美缝合！")

    # 3. 加载数据
    train_set = LBAPDatasetWithSub(args.train_path, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_set  = LBAPDatasetWithSub(args.test_path, split='ood_test')
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 4. 阶段一：在训练集上训练辅助头
    main_model = train_aux_head_only(main_model, train_loader, device, args.adapt_lr, args.adapt_epochs, args.mask_rate)

    # 5. 阶段二：在测试集上 TTA 发威！
    test_perf = eval_with_meta_auxiliary(
    main_model, test_loader, device,
    args.bn_lr, args.aux_lr, args.test_aux_lr,args.inner_steps, args.mask_rate
)

    # 6. 打印成绩
    print("\n" + "="*50)
    print(f"🔥 FINAL TTA AUC:      {test_perf['auc']:.4f}")
    print(f"🎯 FINAL TTA Accuracy: {test_perf['accuracy']:.4f}")
    print("="*50)