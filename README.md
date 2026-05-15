```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 1e-3 \
  --inner_steps 3 \
  --aux_lr 0.01 \
  --bn_lr 0.01 \ 训练时更新主模型bn
  --test_aux_lr 0.01 \
  --mask_rate 0.2 \
  --seed 2022 \
  --device 0
```
[INFO] 正在缝合预训练权重: log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth
✅ 权重完美缝合！
[Dataset] 正在尝试加载大型 JSON 文件: data/ec50/lbap_core_ec50_scaffold_brics.json
[Dataset] 成功加载 train 集，共 2570 个分子。
[Dataset] 正在尝试加载大型 JSON 文件: data/ec50/lbap_core_ec50_scaffold_brics.json
[Dataset] 成功加载 ood_test 集，共 2533 个分子。

[🚀 ADAPTATION] 开始在源域上训练专属辅助探针 (Epochs=5, LR=0.001)...
Adapt Epoch 1/5: 100% 21/21 [00:21<00:00,  1.02s/it]
   -> Epoch 1 Aux Loss: 1.5005
Adapt Epoch 2/5: 100% 21/21 [00:20<00:00,  1.02it/s]
   -> Epoch 2 Aux Loss: 1.0120
Adapt Epoch 3/5: 100% 21/21 [00:20<00:00,  1.02it/s]
   -> Epoch 3 Aux Loss: 0.8689
Adapt Epoch 4/5: 100% 21/21 [00:20<00:00,  1.02it/s]
   -> Epoch 4 Aux Loss: 0.7919
Adapt Epoch 5/5: 100% 21/21 [00:20<00:00,  1.02it/s]
   -> Epoch 5 Aux Loss: 0.7592
✅ 专属辅助探针训练完毕！模型已具备极其敏锐的分布感知能力。

[TTA] 🚀 Meta-Aux TTA (steps=3, lr=0.01, mask_rate=0.2)...
Meta-Aux TTA Testing:   0% 0/20 [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
Meta-Aux TTA Testing: 100% 20/20 [00:20<00:00,  1.02s/it]

🔥 FINAL TTA AUC:      0.5884
🎯 FINAL TTA Accuracy: 0.7071

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 1e-3 \
  --inner_steps 3 \
  --aux_lr 0.01 \
  --bn_lr 0.02 \ 训练时更新主模型bn
  --test_aux_lr 0.01 \
  --mask_rate 0.2 \
  --seed 2022 \
  --device 0
  ```
🔥 FINAL TTA AUC:      0.5908
🎯 FINAL TTA Accuracy: 0.7094

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 1e-3 \
  --inner_steps 3 \
  --aux_lr 0.01 \
  --bn_lr 0.02 \
  --test_aux_lr 0.01 \
  --mask_rate 0.2 \训练时更新主模型全部
  --seed 2022 \
  --device 0
```

🔥 FINAL TTA AUC:      0.6001
🎯 FINAL TTA Accuracy: 0.7114

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 1e-3 \
  --inner_steps 3 \
  --aux_lr 0.01 \训练时去掉aux
  --bn_lr 0.02 \ 训练时更新主模型全部
  --test_aux_lr 0.01 \
  --mask_rate 0.2 \
  --seed 2022 \
  --device 0
```

🔥 FINAL TTA AUC:      0.5964
🎯 FINAL TTA Accuracy: 0.7110

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 1e-3 \
  --inner_steps 3 \
  --aux_lr 0.01 \训练时去掉aux
  --bn_lr 0.02 \训练时更新主模型bn
  --test_aux_lr 0.01 \
  --mask_rate 0.2 \
  --seed 2022 \
  --device 0
```
🔥 FINAL TTA AUC:      0.5853
🎯 FINAL TTA Accuracy: 0.7067

## 说明更新主模型全部比bn更好，能高1 下面都更新主模型全部

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 1e-3 \
  --inner_steps 3 \
  --aux_lr 0.01 \
  --bn_lr 0.02 \
  --test_aux_lr 0.02 \
  --mask_rate 0.2 \
  --seed 2022 \
  --device 0
```
🔥 FINAL TTA AUC:      0.6010
🎯 FINAL TTA Accuracy: 0.7118

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 1e-3 \
  --inner_steps 3 \
  --aux_lr 0.02 \
  --bn_lr 0.02 \
  --test_aux_lr 0.02 \
  --mask_rate 0.2 \
  --seed 2022 \
  --device 0
```
🔥 FINAL TTA AUC:      0.6007
🎯 FINAL TTA Accuracy: 0.7118

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 0.01 \
  --inner_steps 3 \
  --aux_lr 0.01 \
  --bn_lr 0.02 \
  --test_aux_lr 0.02 \
  --mask_rate 0.2 \
  --seed 2022 \
  --device 0
```
🔥 FINAL TTA AUC:      0.6060
🎯 FINAL TTA Accuracy: 0.7138

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 0.01 \
  --inner_steps 1 \轮数下降也不行
  --aux_lr 0.01 \
  --bn_lr 0.02 \
  --test_aux_lr 0.02 \
  --mask_rate 0.2 \
  --seed 2022 \
  --device 0
```
🔥 FINAL TTA AUC:      0.6023
🎯 FINAL TTA Accuracy: 0.7043

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 0.01 \
  --inner_steps 5 \ 轮数上升也不行
  --aux_lr 0.01 \
  --bn_lr 0.02 \
  --test_aux_lr 0.02 \
  --mask_rate 0.2 \
  --seed 2022 \
  --device 0
```
🔥 FINAL TTA AUC:      0.5989
🎯 FINAL TTA Accuracy: 0.7134

-----------------改成一致性正则化--------------
```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 0.01 \
  --inner_steps 3 \
  --aux_lr 0.01 \
  --bn_lr 1e-4 \
  --test_aux_lr 0.02 \
  --mask_rate 0.1 \
  --seed 2022 \
  --device 0
```
<img width="1005" height="753" alt="image" src="https://github.com/user-attachments/assets/9395e3fa-dbaf-490c-badb-3934138eef8a" />


mask_rate 0.2


<img width="858" height="247" alt="image" src="https://github.com/user-attachments/assets/7b31de40-355d-437d-a048-e315b4ba3866" />
aux_lr=1e-3
<img width="862" height="229" alt="image" src="https://github.com/user-attachments/assets/61b75450-2e13-42f0-b14a-fa9199685a5e" />

```
!python evaluate_drugtta.py \
  --model_path log/lbap_core_ec50_scaffold_brics/PyG_GIN/best_model.pth \
  --train_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --test_path data/ec50/lbap_core_ec50_scaffold_brics.json \
  --batch_size 128 \
  --adapt_epochs 5 \
  --adapt_lr 0.01 \
  --inner_steps 5 \
  --aux_lr 0.005 \
  --bn_lr 1e-4 \
  --mask_rate 0.2 \
  --aux_weight 0.2 \
  --seed 2022 \
  --device 0
```
<img width="718" height="190" alt="image" src="https://github.com/user-attachments/assets/da8c4096-146f-4511-a481-f5f3a05e050a" />

<img width="1394" height="774" alt="image" src="https://github.com/user-attachments/assets/ac91a2f4-421f-4f2e-b033-0b48d8ed60a2" />

