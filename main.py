import os
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from models.tta import eval_with_tent
import torch
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
    loss_curv, min_loss, best_ep, best_para = [], None, None, {}
    
    for ep in range(args.epoch_ast):
        print(f'\n[INFO] Training Assistant Models - Epoch {ep}')
        conditional_gnn.train()
        domain_classifier.train()
        Eqs, ELs = [], []
        
        for data in tqdm(train_loader):
            data = data.to(device) # PyG 自动把图、标签等全搬到 GPU
            batch_size = data.y.size(0)
            
            # 推断环境分布 q_e
            q_e = torch.softmax(domain_classifier(data), dim=-1)
            losses = []
            
            for dom in range(args.num_domain):
                domain_info = torch.ones(batch_size, dtype=torch.long, device=device) * dom
                # 假想敌预测
                p_ye = conditional_gnn(data, domain_info)
                # 【关键修改】data.y 是标签
                loss = cross_entropy(p_ye, data.y.view(-1), reduction='none')
                losses.append(loss)
                
            losses = torch.stack(losses, dim=1)
            Eq = torch.mean(torch.sum(q_e * losses, dim=-1))
            ELBO = Eq + KLDist(q_e, prior)

            optimizer_con.zero_grad()
            optimizer_dom.zero_grad()
            ELBO.backward()
            optimizer_con.step()
            optimizer_dom.step()

            Eqs.append(Eq.item())
            ELs.append(ELBO.item())
            
        mean_ELBO = np.mean(ELs)
        print(f'[INFO] Eq: {np.mean(Eqs):.4f}, ELBO: {mean_ELBO:.4f}')
        loss_curv.append((np.mean(Eqs), mean_ELBO))
        
        if best_ep is None or mean_ELBO < min_loss:
            min_loss, best_ep = mean_ELBO, ep
            best_para = {
                'con': deepcopy(conditional_gnn.state_dict()),
                'dom': deepcopy(domain_classifier.state_dict())
            }

    print(f'[INFO] Using the best Assistant Model from Epoch {best_ep}')
    domain_classifier.load_state_dict(best_para['dom'])
    conditional_gnn.load_state_dict(best_para['con'])
    domain_classifier.eval()
    conditional_gnn.eval()

    # ==========================================
    # 5. 阶段二：训练主模型 (学习不变性)
    # ==========================================
    valid_curv, test_curv, max_valid_auc = {}, {}, None
    model_path = os.path.join(log_dir, f'best_model.pth')
    
    for ep in range(args.epoch_main):
        print(f'\n[INFO] Training Main Model - Epoch {ep}')
        main_model.train()
        
        for data in tqdm(train_loader):
            data = data.to(device)
            batch_size = data.y.size(0)
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

            # 主角盲猜
            pred = main_model(data)
            
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

        print("\n--- TTA 评估 (TENT Evaluation) ---")
        # 传入原来的 evaluate 函数
        from models.utils import evaluate 
        test_perf_tta = eval_with_tent(
            main_model, 
            test_loader, 
            device, 
            tta_steps=1, 
            tta_lr=1e-3, 
            evaluate_fn=evaluate
        )

        print(f'[INFO] TENT Test : {test_perf_tta}')
        if max_valid_auc is None or val_perf['auc'] > max_valid_auc:
            torch.save({
                'main': main_model.state_dict(),
                'dom': best_para['dom'],
                'con': best_para['con']
            }, model_path)
            max_valid_auc = val_perf['auc']

    # 打印最终结果
    best_result = {}
    for k, v in valid_curv.items():
        pos = int(np.argmax(v))
        best_result[k] = [pos, v[pos], test_curv[k][pos]]
    print(f'\n[🎉 FINAL RESULT] Best Test performance based on Valid set:')
    print(best_result)