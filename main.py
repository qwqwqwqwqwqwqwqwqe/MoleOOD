import os
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from models.tta import eval_with_tent
import torch
import math
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch_geometric.loader import DataLoader  # 【关键修改】使用 PyG 的 DataLoader
from torch_geometric.seed import seed_everything
from models.tta import eval_with_shot
# 导入我们新写的纯净版 Dataset
from models.dataset import LBAPDatasetWithSub 

# 导入原有的损失函数和评估逻辑 (假设这些纯数学函数不需要改)
from models import KLDist, MeanLoss, DeviationLoss, discrete_gaussian, evaluate
# 导入模型 (注意：下一步你需要修改这些模型以接收 PyG 数据)
from models import Framework, ConditionalGnn, DomainClassifier


def init_args():
    parser = argparse.ArgumentParser('Experiment for Drugood Dataset (PyG Version)')
    
    # --- 【关键修改】去掉了 data_config，直接在这里指定数据路径 ---
    parser.add_argument('--train_path', type=str, default='data/ec50/lbap_core_ec50_assay_brics.json')
    parser.add_argument('--val_path', type=str, default='data/ec50/lbap_core_ec50_assay_brics.json') # 注意：原配置里 val 和 test 可能在同一个大 json 里
    parser.add_argument('--test_path', type=str, default='data/ec50/lbap_core_ec50_assay_brics.json')
    
    # 为了替代 model_config，提取出核心维度参数
    parser.add_argument('--emb_dim', default=128, type=int, help='GNN hidden dimension')
    parser.add_argument('--num_class', default=2, type=int, help='number of classes')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)

    # 训练超参数 (保持不变)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--num_domain', default=20, type=int)
    parser.add_argument('--epoch_main', default=50, type=int)
    parser.add_argument('--epoch_ast', default=20, type=int)
    parser.add_argument('--lambda_loss', default=1.0, type=float)
    parser.add_argument('--dist', default='uniform', type=str)
    
    return parser.parse_args()


def make_log(args):
    dataset_name = os.path.basename(args.train_path).split('.')[0]
    log_dir = os.path.join('log', dataset_name, 'PyG_GIN')
    os.makedirs(log_dir, exist_ok=True)
    fname = f'seed_{args.seed}-dom_{args.num_domain}-ast_{args.epoch_ast}-main_{args.epoch_main}.json'
    return log_dir, fname


def get_prior(num_domain, dtype='uniform'):
    if dtype == 'uniform':
        return torch.ones(num_domain) / num_domain
    else:
        return discrete_gaussian(num_domain)


def eval_one_epoch(model, loader, device, verbose=True):
    model.eval()
    result_all, gt_all = [], []
    
    for data in (tqdm(loader) if verbose else loader):
        # 【关键修改】PyG 的数据转移只需要一句 data.to(device)
        data = data.to(device)
        
        with torch.no_grad():
            # 【关键修改】模型直接接收 data 对象。子结构已经在 data.subs 里了
            logits = model(data)
            result = torch.softmax(logits, dim=-1)
            
            result_all.append(result.detach().cpu())
            # 【关键修改】标签现在叫 data.y
            gt_all.append(data.y.cpu().view(-1))
            
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])


if __name__ == '__main__':
    args = init_args()
    print(args)
    seed_everything(args.seed)

    log_dir, log_name = make_log(args)
    device = torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}')

    # ==========================================
    # 1. 数据集构建 (调用新的 dataset.py 和 PyG DataLoader)
    # ==========================================
    print("[INFO] Building Datasets...")
    train_set = LBAPDatasetWithSub(args.train_path, split='train')
    valid_set = LBAPDatasetWithSub(args.val_path, split='ood_val')
    test_set  = LBAPDatasetWithSub(args.test_path, split='ood_test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ==========================================
    # 2. 模型构建 (去掉了 mmcv 配置，直接传参)
    # ==========================================
    print("[INFO] Building Models...")
    # 注意：这里的模型内部实现（如 Framework）下一步需要修改为接收 PyG 数据
    main_model = Framework(
        dropout=args.dropout, base_dim=args.emb_dim, sub_dim=args.emb_dim, num_class=args.num_class
    ).to(device)
    
    conditional_gnn = ConditionalGnn(
        emb_dim=args.emb_dim, backend_dim=args.emb_dim, num_domain=args.num_domain, num_class=args.num_class
    ).to(device)
    
    domain_classifier = DomainClassifier(
        backend_dim=args.emb_dim, num_task=1, num_domain=args.num_domain
    ).to(device)

    # =====================================================================
    # 🚀 【核心新增】：加载无辅助任务版本的预训练模型 (Warm-start)
    # =====================================================================
    # 请通过 argparse 传入，或者在这里硬编码你之前跑出来的 best_model.pth 的路径
    pretrained_path = 'log/lbap_core_ec50_scaffold_brics/PyG_GIN/pretrained_ec50_scaffold.pth' # 替换为你的真实路径
    
    if os.path.exists(pretrained_path):
        print(f"\n[🚀 WARM-START] 正在加载预训练的最强 MoleOOD 模型: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # 1. 提取预训练的主模型参数
        pretrained_dict = checkpoint['main']
        
        # 2. 获取当前我们带有 aux_predictor 的新模型的状态字典
        model_dict = main_model.state_dict()
        
         # 1. 过滤掉完全不相关的层 (比如新加的 aux_predictor)
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # 2. 🚨 【外科手术核心】：处理维度不匹配的层 (如 node_embeddings)
        for k, v in filtered_dict.items():
            # 检查旧参数和新模型中对应参数的形状是否一致
            if v.shape != model_dict[k].shape:
                print(f"[Surgery] 检测到形状不匹配的层: {k} (旧: {v.shape} -> 新: {model_dict[k].shape})")
                
                # 情况 A：如果这是一个权重矩阵 [out_features, in_features] 且 in_features 扩展了
                # 比如 [128, 39] -> [128, 40]
                if len(v.shape) == 2 and v.shape[0] == model_dict[k].shape[0] and v.shape[1] < model_dict[k].shape[1]:
                    # 创建一个新的空壳，形状等于新模型的形状
                    new_w = torch.empty_like(model_dict[k])
                    # 用标准的 Kaiming 初始化填满整个新壳 (保证第 40 列有合理的初始值)
                    torch.nn.init.kaiming_uniform_(new_w, a=math.sqrt(5))
                    # 将旧模型的参数，精准地覆盖到新壳的“前半部分”
                    new_w[:, :v.shape[1]] = v
                    # 把缝合好的新权重放回字典
                    filtered_dict[k] = new_w
                    print(f"  -> [成功缝合] 旧权重已复制，新增的列已随机初始化。")
                else:
                    # 如果是其他未知的形状不匹配（比如你改了隐藏层维度），直接跳过不加载
                    print(f"  -> [跳过加载] 形状差异无法自动缝合，保持新模型的随机初始化。")
                    filtered_dict[k] = model_dict[k]
                    
        # 3. 用缝合好的参数更新当前模型的状态字典
        model_dict.update(filtered_dict)
        
        # 5. 正式加载进去！(strict=False 表示允许有不匹配的 key)
        main_model.load_state_dict(model_dict, strict=False)
        
        # 可选：如果你希望连假想敌和推断器的经验也一起继承
        domain_classifier.load_state_dict(checkpoint['dom'])
        conditional_gnn.load_state_dict(checkpoint['con'])
        
        print("[🚀 WARM-START] 预训练模型加载成功！GNN 骨干已注入前世记忆。\n")
    else:
        print(f"\n[⚠️ WARNING] 未找到预训练模型 {pretrained_path}，将从零开始随机初始化训练。\n")
    # =====================================================================

    # ==========================================
    # 3. 优化器与损失函数
    # ==========================================
    optimizer_main = AdamW(main_model.parameters(), lr=args.lr)
    optimizer_dom = AdamW(domain_classifier.parameters(), lr=args.lr)
    optimizer_con = AdamW(conditional_gnn.parameters(), lr=args.lr)
    
    CLSLoss = torch.nn.CrossEntropyLoss()
    mean_loss = MeanLoss(CLSLoss)
    dev_loss = DeviationLoss(activation='abs', reduction='mean')
    prior = get_prior(args.num_domain, args.dist).to(device)

    # ==========================================
    # 4. 阶段一：训练辅助模型 (推断环境)
    # ==========================================
    # loss_curv, min_loss, best_ep, best_para = [], None, None, {}
    
    # for ep in range(args.epoch_ast):
    #     print(f'\n[INFO] Training Assistant Models - Epoch {ep}')
    #     conditional_gnn.train()
    #     domain_classifier.train()
    #     Eqs, ELs = [], []
        
    #     for data in tqdm(train_loader):
    #         data = data.to(device) # PyG 自动把图、标签等全搬到 GPU
    #         batch_size = data.y.size(0)
            
    #         # 推断环境分布 q_e
    #         q_e = torch.softmax(domain_classifier(data), dim=-1)
    #         losses = []
            
    #         for dom in range(args.num_domain):
    #             domain_info = torch.ones(batch_size, dtype=torch.long, device=device) * dom
    #             # 假想敌预测
    #             p_ye = conditional_gnn(data, domain_info)
    #             # 【关键修改】data.y 是标签
    #             loss = cross_entropy(p_ye, data.y.view(-1), reduction='none')
    #             losses.append(loss)
                
    #         losses = torch.stack(losses, dim=1)
    #         Eq = torch.mean(torch.sum(q_e * losses, dim=-1))
    #         ELBO = Eq + KLDist(q_e, prior)

    #         optimizer_con.zero_grad()
    #         optimizer_dom.zero_grad()
    #         ELBO.backward()
    #         optimizer_con.step()
    #         optimizer_dom.step()

    #         Eqs.append(Eq.item())
    #         ELs.append(ELBO.item())
            
    #     mean_ELBO = np.mean(ELs)
    #     print(f'[INFO] Eq: {np.mean(Eqs):.4f}, ELBO: {mean_ELBO:.4f}')
    #     loss_curv.append((np.mean(Eqs), mean_ELBO))
        
    #     if best_ep is None or mean_ELBO < min_loss:
    #         min_loss, best_ep = mean_ELBO, ep
    #         best_para = {
    #             'con': deepcopy(conditional_gnn.state_dict()),
    #             'dom': deepcopy(domain_classifier.state_dict())
    #         }

    # print(f'[INFO] Using the best Assistant Model from Epoch {best_ep}')
    # domain_classifier.load_state_dict(best_para['dom'])
    # conditional_gnn.load_state_dict(best_para['con'])
    # domain_classifier.eval()
    # conditional_gnn.eval()

    # ==========================================
    # 5. 阶段二：训练主模型 (学习不变性)
    # ==========================================
    valid_curv, test_curv, max_valid_auc = {}, {}, None
    model_path = os.path.join(log_dir, f'best_model.pth')
    from models.utils import mask_node_features
    # --- 【核心新增 1】：分离参数，配置元学习的“内循环优化器” ---
    # 我们需要一个只更新 BN 层和 辅助分类头 的优化器
    bn_and_aux_params = []
    for name, param in main_model.named_parameters():
        # 筛选出所有的 BatchNorm 层参数，以及我们新加的 aux_predictor 参数
        if 'bn' in name or 'batch_norm' in name or 'aux_predictor' in name:
            bn_and_aux_params.append(param)

    # 论文建议内循环学习率 (α) 通常与主学习率相似或略大
    optimizer_inner = AdamW(bn_and_aux_params, lr=5e-4) 

    # optimizer_main (外循环优化器) 依然负责全体参数，保持不变
    for ep in range(args.epoch_main):
        print(f'\n[INFO] Training Main Model - Epoch {ep}')
        main_model.train()
        
        for data in tqdm(train_loader):
            data = data.to(device)
            batch_size = data.y.size(0)

            # =====================================================================
            # 🚀 Step 1: 内循环 (Inner Loop) - 辅助任务预热 BN 层
            # 对应 Drug-TTA 论文 Fig 2(c) Step 1 & 2
            # =====================================================================
            # 1. 制造辅助任务的输入 (遮挡 15% 的原子)
            masked_data, mask_idx, masked_labels = mask_node_features(data)
            
            if mask_idx is not None:
                # 2. 模型执行辅助预测 (此时必须传 return_node_feats=True)
                # 注意：你在 Framework 的 forward 里必须实现这个逻辑！
                aux_logits = main_model(masked_data, return_node_feats=True)
                
                # 3. 计算辅助任务 Loss (只算被遮挡节点)
                aux_loss = F.cross_entropy(aux_logits[mask_idx], masked_labels)
                
                # 4. 反向传播，只让 optimizer_inner 去更新 BN 和 Aux 头！
                # 主干网络 (GNN) 的卷积参数绝对不会动！
                optimizer_inner.zero_grad()
                aux_loss.backward()
                optimizer_inner.step() 
            
            # =====================================================================
            # 🚀 Step 2: 外循环 (Outer Loop) - 主任务学习不变性
            # 对应 Drug-TTA 论文 Fig 2(c) Step 3 & 4
            # 此时，模型里的 BN 层已经刚刚被辅助任务“预热”过了！
            # =====================================================================
            cond_result = []
            
            with torch.no_grad():
                # 计算假想敌期望得分
                for dom in range(args.num_domain):
                    domain_info = torch.ones(batch_size, dtype=torch.long, device=device) * dom
                    cond_term = cross_entropy(
                        conditional_gnn(data, domain_info), 
                        data.y.view(-1), reduction='none'
                    )
                    cond_result.append(cond_term)
                cond_result = torch.stack(cond_result, dim=0)
                cond_result = torch.matmul(prior, cond_result)

                # 计算推断的伪标签
                p_e = domain_classifier(data)
                group = torch.argmax(p_e, dim=-1)

            # (MoleOOD 原逻辑不变) 主角盲猜 (使用原始的、未被遮挡的 data!)
            # 此时必须传 return_node_feats=False
            pred = main_model(data, return_node_feats=False)
            
            mean_term = mean_loss(pred, data.y.view(-1), group)
            this_loss = cross_entropy(pred, data.y.view(-1), reduction='none')
            
            dev_term = dev_loss(this_loss, cond_result)
            loss = args.lambda_loss * mean_term + dev_term
            
            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()

        # ==========================================
        # 6. 评估与保存
        # ==========================================
        print("\n--- 标准评估 (Standard Evaluation) ---")
        val_perf = eval_one_epoch(main_model, valid_loader, device)
        test_perf = eval_one_epoch(main_model, test_loader, device)
        
        for k, v in val_perf.items():
            if k not in valid_curv:
                valid_curv[k], test_curv[k] = [], []
            valid_curv[k].append(val_perf[k])
            test_curv[k].append(test_perf[k])

        print(f'[INFO] Valid: {val_perf}')
        print(f'[INFO] Test : {test_perf}')

        
        if max_valid_auc is None or val_perf['auc'] > max_valid_auc:
            torch.save({
                'main': main_model.state_dict(),
                # 这个ec50_scaffold有训练好的dom,con，前面训练部分注释了没有best_para直接加载
                # 训练其他模型时没有预训练的模型，要改回来
                # 'dom': best_para['dom'],
                # 'con': best_para['con']
                'dom': domain_classifier.state_dict(), # 直接存当前状态
                'con': conditional_gnn.state_dict()    # 直接存当前状态
            }, model_path)
            max_valid_auc = val_perf['auc']
        

    # 打印最终结果
    best_result = {}
    for k, v in valid_curv.items():
        pos = int(np.argmax(v))
        best_result[k] = [pos, v[pos], test_curv[k][pos]]
    print(f'\n[🎉 FINAL RESULT] Best Test performance based on Valid set:')
    print(best_result)