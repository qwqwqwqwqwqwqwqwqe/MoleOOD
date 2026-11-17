# 版本 A：单个共享节点特征 Prompt

本实验是 GPF 思路在现有 MoleOOD 上的适配，不是原论文的完整复现。没有 L2P 池、key 或路由。

主图节点 embedding 加同一个 128 维向量，再经过原 GIN；子结构均值聚合及分类头保持原样。准备阶段只学习辅助头和提示，目标阶段仅更新提示。所有 BN buffer 固定，dropout 关闭。

## 环境与数据

使用现有能运行 MoleOOD 的服务器环境（PyTorch、PyG、torch-scatter、RDKit、scikit-learn、numpy，测试另需 pytest）。本地 `.venv` 不一定可用。不要直接覆盖已有服务器环境；如新建环境，请按服务器 PyTorch/CUDA 版本安装对应的 PyG 扩展。

JSON 使用原工程格式：`split.train` 和 `split.ood_test`，样本包含 `smiles`、`cls_label`、`domain_id` 和可选 `substructure`。准备与测试可指向同一个 JSON，因为读取的是不同 split。脚本会按规范化 SMILES 检查源/目标重叠并报错；不要绕过它来追求高 AUC。

默认使用 `log/lbap_core_ec50_scaffold_brics/PyG_GIN/pretrained_ec50_scaffold.pth`。39→40 输入扩展新增 MASK 列置零，其他形状不匹配直接报错；不会静默随机替换主干。原有原子序数截断编码暂时保留，以兼容 checkpoint。

## 服务器运行

以下均在项目根目录执行，修改数据路径；命令使用 Linux shell。

```bash
python -m pytest tests/test_prompt_a.py -q

python prompt_a.py \
  --stage all \
  --model-path log/lbap_core_ec50_scaffold_brics/PyG_GIN/pretrained_ec50_scaffold.pth \
  --train-path /path/to/lbap_core_ec50_scaffold_brics.json \
  --test-path /path/to/lbap_core_ec50_scaffold_brics.json \
  --device cuda:0 --seed 2022 --output log/prompt_a/seed2022
```

脚本分三个阶段：

1. 源域按规范化分子身份分组，确定性划出约 10% 作为准备验证集。它可能已被原主干训练见过，因此不是全新泛化测试集。
2. 主干固定，辅助头预热最多 5 epoch；固定辅助头后，提示用源域分类损失＋0.5×重建损失训练最多 5 epoch，以准备验证损失选 checkpoint。验证掩码固定。
3. 在原始 `ood_test` 顺序上比较四组，不使用测试标签调参或选择权重。

| 结果名 | 含义 |
| --- | --- |
| moleood | 零提示、不更新；辅助头准备不改变该预测路径 |
| source_frozen | 源域准备后的提示，目标阶段不更新 |
| episodic | 每批恢复准备后的提示，并清空在线 Adam 状态 |
| continual | 跨 batch 保留提示及在线 Adam 状态 |

所有适应均先移除属性标签和环境编号，使用掩码重建及 KL 一致性，再预测干净图。教师是本批更新前同一提示的干净图输出，不是真实标签。单步更新使用梯度裁剪和 batch 更新范数上限 0.05；额外正则锚定源域准备后的提示，不是本步起点。

可分别运行准备和评估：

```bash
python prompt_a.py --stage prepare \
  --model-path log/lbap_core_ec50_scaffold_brics/PyG_GIN/pretrained_ec50_scaffold.pth \
  --train-path /path/to/lbap_core_ec50_scaffold_brics.json \
  --device cuda:0 --output log/prompt_a/prepared

python prompt_a.py --stage evaluate \
  --model-path log/lbap_core_ec50_scaffold_brics/PyG_GIN/pretrained_ec50_scaffold.pth \
  --prepared-path log/prompt_a/prepared/prepared.pt \
  --test-path /path/to/lbap_core_ec50_scaffold_brics.json \
  --device cuda:0 --output log/prompt_a/evaluation
```

`--help` 列出损失权重、学习率、步数和范数限制。先固定默认配置跑 2022/2023/2024 三个种子，各种子独立输出目录。不要根据 `ood_test` 挑最好的一组超参数。

## 输出与判断

输出目录包含 `prepared.pt`、`results.json`、各组逐样本预测 NPZ、逐批损失/提示范数 JSON，以及最终提示向量。结果保存初始 checkpoint SHA256、配置、源域训练历史；prepared.pt 额外保存源域身份及准备划分。目标只做固定顺序评估，尚未包含断点续跑、环境重复流或历史环境遗忘矩阵。

`auc_gain_pp` 是相对原 MoleOOD 的 AUC 百分点差。单类别测试集 AUC 为 null。先检查零提示基线，再比较 source_frozen 与 moleood，最后看 episodic/continual 是否优于 source_frozen。只有源域准备收益不能证明 TTA 有效；只有重建下降不能证明性质分类改善。

此实现的重建损失按有效分子平均，小于两个节点的分子跳过重建。四组使用相同目标顺序，两个适应组共享相同掩码种子。每组结束都会断言模型全部权重和 buffer 未改变。真实 PyG 测试检查零提示等价与 Prompt 梯度；缺少图依赖时该项会跳过，不能视为完整集成验证通过。

## Google Colab

打开仓库 `prompt-a` 分支中的 `colab_prompt_a.ipynb`。该分支已包含 EC50 scaffold 的 `best_model.pth`，notebook 使用它作为统一基线；本文前面的 `pretrained_ec50_scaffold.pth` 是本地权重名，不在远程原分支中，直接 clone 时请用 notebook 的路径。不要混用不同 checkpoint 比较结果。将 JSON 放入 Drive 并修改 DATA；默认将结果保存回 Drive。依赖安装按当前 PyTorch/CUDA 选择 PyG wheel，仅允许二进制安装，不自动从源码编译。
