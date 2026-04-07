import os
import torch

import torch.nn as nn
from copy import deepcopy
import argparse
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from tqdm import tqdm 
import torch.nn.functional as F # 用于熵计算可能用到的 F.softmax 等
# 导入我们重构后的核心组件
from models.dataset import LBAPDatasetWithSub
from models.Framework import Framework, ConditionalGnn, DomainClassifier # 假设模型类都在这个文件里
from models.utils import evaluate

# 导入我们新加的 TTA 模块 (如果需要的话)
# from models.tta import eval_with_tent

def init_args():
    parser = argparse.ArgumentParser('Evaluation Script for Pre-trained MoleOOD Model (PyG Version)')
    
    # --- 核心参数：指定要加载的模型和要测试的数据 ---
    parser.add_argument(
        '--model_path', type=str, required=True,
        help="Path to the trained .pth model file (e.g., log/.../best_model.pth)"
    )
    parser.add_argument(
        '--test_path', type=str, required=True,
        help="Path to the test data JSON file (e.g., data/ic50/lbap_core_ic50_scaffold_brics.json)"
    )
    
    # --- 模型架构参数 (必须与训练时完全一致！) ---
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--dropout', default=0.1, type=float) # 注意：这里是MyGIN的dropout
    parser.add_argument('--framework_dropout', default=0.5, type=float) # Framework的predictor dropout
    parser.add_argument('--num_domain', default=20, type=int)

    # --- 数据加载参数 ---
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--device', default=0, type=int)
    
    return parser.parse_args()


def eval_one_epoch(model, loader, device, verbose=True):
    """
    这是一个简化的评估函数，与 main.py 中的版本一致
    """
    model.eval()
    result_all, gt_all = [], []
    
    for data in (tqdm(loader) if verbose else loader):
        data = data.to(device)
        with torch.no_grad():
            logits = model(data)
            result = torch.softmax(logits, dim=-1)
            
            result_all.append(result.detach().cpu())
            gt_all.append(data.y.cpu().view(-1))
            
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'accuracy'])


if __name__ == '__main__':
    args = init_args()
    print("--- Evaluation Arguments ---")
    print(args)

    device = torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}')

    # 1. 构建一个与训练时【架构完全相同】的空模型
    print("\n[INFO] Building model architecture...")
    main_model = Framework(
        base_dim=args.emb_dim, 
        sub_dim=args.emb_dim, 
        num_class=args.num_class, 
        dropout=args.framework_dropout
    ).to(device)

    # 2. 加载预训练的权重
    print(f"\n[INFO] Loading pre-trained weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    # 加载包含 'main', 'dom', 'con' 的字典
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 只需要加载主角模型 'main' 的参数
    main_model.load_state_dict(checkpoint['main'])
    print("✅ Weights loaded successfully!")

    # 3. 准备测试数据集
    print(f"\n[INFO] Loading test data from: {args.test_path}")
    test_set  = LBAPDatasetWithSub(args.test_path, split='ood_test')
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 4. 执行评估
    print("\n[INFO] Running evaluation on the test set...")
    test_perf = eval_one_epoch(main_model, test_loader, device)

    # 5. 打印最终结果
    print("\n" + "="*50)
    print("🎉 FINAL EVALUATION RESULT 🎉")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.test_path}")
    print(f"AUC: {test_perf['auc']:.4f}")
    print(f"Accuracy: {test_perf['accuracy']:.4f}")
    print("="*50)

    # --- (可选) 如果你想在这里直接测试 TTA ---
    from models.tta import eval_with_tent
    print("\n[INFO] Running Test-Time Adaptation (TENT) evaluation...")
    test_perf_tta = eval_with_tent(main_model, test_loader, device, evaluate_fn=evaluate)
    print("\n--- TTA RESULT ---")
    print(f"TENT AUC: {test_perf_tta['auc']:.4f}")
    print(f"TENT Accuracy: {test_perf_tta['accuracy']:.4f}")