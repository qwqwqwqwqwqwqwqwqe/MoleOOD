跑了EC50-Scaffold

<img width="993" height="911" alt="image" src="https://github.com/user-attachments/assets/e948cbaa-002a-4eed-8952-ac5b1e26cf62" />


[🎉 FINAL RESULT] Best Test performance based on Valid set:
{'auc': [34, np.float64(0.7164741024720755), np.float64(0.6690684309504368)], 'accuracy': [41, 0.759873628616333, 0.7149624824523926]}

论文中是66.69+-0.34
<img width="1007" height="381" alt="image" src="https://github.com/user-attachments/assets/1cc8c0be-a2c5-458a-8936-81a52d520193" />

EC50-Assay
<img width="1479" height="861" alt="image" src="https://github.com/user-attachments/assets/5cfca458-7bb3-42d8-9006-b10d030ef062" />

EC50-Size
<img width="1295" height="945" alt="image" src="https://github.com/user-attachments/assets/068177a4-5007-45a5-8d50-39fe5d03948d" />
没在65.09+-0.9
加了tent的结果
<img width="1286" height="849" alt="image" src="https://github.com/user-attachments/assets/85cc9007-ddc3-4f92-bdf5-1cf867d9c3d9" />

加了基于聚类shot的结果
<img width="1244" height="867" alt="image" src="https://github.com/user-attachments/assets/9d1c77f2-5875-4d00-be98-cc46fba120de" />

## 加了drugtta的结果
## 元学习微调：
!python main.py \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --val_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --epoch_ast 0 \
  --epoch_main 20 \
  --lr 1e-4 \
  --num_domain 20 \
  --device 0 \
  --seed 42
  
[🎉 FINAL RESULT] Best Test performance based on Valid set:
{'auc': [12, np.float64(0.7116385955844049), np.float64(0.6510927022245316)], 'accuracy': [18, 0.730252742767334, 0.6936439275741577]

## 测试时
[INFO] Loading test data from: data/ec50/lbap_core_ec50_scaffold_brics.json
[Dataset] 正在尝试加载大型 JSON 文件: data/ec50/lbap_core_ec50_scaffold_brics.json
[Dataset] 成功加载 ood_test 集，共 2533 个分子。

[TTA] 🚀 启动 Meta-Auxiliary TTA (steps=1, lr=0.005, mask_rate=0.15)...
Meta-Aux TTA Testing: 100% 20/20 [00:12<00:00,  1.63it/s]


Model: log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth

Test Dataset: data/ec50/lbap_core_ec50_scaffold_brics.json

TTA LR: 0.005 | TTA Steps: 1 | Mask Rate: 0.15
--------------------------------------------------
🔥 AUC:      0.6479
🎯 Accuracy: 0.7015

TTA LR: 0.01 | TTA Steps: 1 | Mask Rate: 0.4
--------------------------------------------------
🔥 AUC:      0.6550
🎯 Accuracy: 0.7055

TTA LR: 0.01 | TTA Steps: 3 | Mask Rate: 0.4
--------------------------------------------------
🔥 AUC:      0.6612

TTA LR: 0.005 | TTA Steps: 1 | Mask Rate: 0.3
--------------------------------------------------
🔥 AUC:      0.6532
🎯 Accuracy: 0.7031

TTA LR: 0.0001 | TTA Steps: 1 | Mask Rate: 0.3
--------------------------------------------------
🔥 AUC:      0.6511
🎯 Accuracy: 0.7000

TTA LR: 0.005 | TTA Steps: 1 | Mask Rate: 0.5
--------------------------------------------------
🔥 AUC:      0.6529
🎯 Accuracy: 0.7004

## 直接用MoleOOD auc=0.6691的模型做元学习测试，不进行辅助头和模型的训练

TTA LR: 0.01 | TTA Steps: 3 | Mask Rate: 0.25
--------------------------------------------------
🔥 AUC:      0.6677
🎯 Accuracy: 0.7315

TTA LR: 0.01 | TTA Steps: 3 | Mask Rate: 0.2
--------------------------------------------------
🔥 AUC:      0.6698
🎯 Accuracy: 0.7300

TTA LR: 0.01 | TTA Steps: 2 | Mask Rate: 0.2
--------------------------------------------------
🔥 AUC:      0.6686
🎯 Accuracy: 0.7323

TTA LR: 0.01 | TTA Steps: 4 | Mask Rate: 0.2
--------------------------------------------------
🔥 AUC:      0.6657
🎯 Accuracy: 0.7268

TTA LR: 0.005 | TTA Steps: 3 | Mask Rate: 0.2
--------------------------------------------------
🔥 AUC:      0.6672
🎯 Accuracy: 0.7308

TTA LR: 0.02 | TTA Steps: 3 | Mask Rate: 0.2
--------------------------------------------------
🔥 AUC:      0.6734
🎯 Accuracy: 0.7284
