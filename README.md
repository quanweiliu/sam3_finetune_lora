# SAM3-LoRA: 基于低秩自适应 (LoRA) 的高效微调

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**仅需训练约 1% 的参数即可微调 SAM3 分割模型**

[快速开始](#快速开始) • [项目架构](#项目架构) • [训练指南](#训练指南) • [推理使用](#推理使用) • [实际案例](#实际案例混凝土裂缝检测) • [配置说明](#配置说明)

</div>

---

## 项目概述 (Overview)

本项目旨在使用 **LoRA (Low-Rank Adaptation)** 技术对 SAM3 (Segment Anything Model 3) 进行微调。LoRA 是一种参数高效的微调方法，可以将可训练参数量从 100% 降低到约 1%，同时保持甚至提升模型在特定任务上的性能。

### 更新 (Updates)

2026-05-01:
- 新增 
  - lqw/sam3_finetune_lora/scripts/data/vaihingen_dataset.py 支持一种新的遥感数据加载
  - 模型支持只有 mask 的 concept prompt segmentation, 通过 mask 直接推理 bbox

- 拆分出数据和一些辅助函数：
- 拆分
  - lqw/sam3_finetune_lora/scripts/training/train_sam3_lora_native.py
  - lqw/sam3_finetune_lora/scripts/inference/validate_sam3_lora.py
  - lqw/sam3_finetune_lora/scripts/inference/infer_sam.py
- 数据
  - 拆分到 lqw/sam3_finetune_lora/scripts/data/COCODataset.py and lqw/sam3_finetune_lora/scripts/data/vaihingen_dataset.py
- 辅助函数
  - 拆分到 /home/icclab/Documents/lqw/sam3_finetune_lora/scripts/tools/utils.py


### 为什么选择本项目？

- ✅ **消费级 GPU 友好**：仅需 16GB 显存即可训练，无需 A100 (80GB)。
- ✅ **极小的权重文件**：LoRA 权重仅 10-50MB，远小于原始模型的 3GB。
- ✅ **快速迭代**：更少的内存占用带来更快的训练速度。
- ✅ **简单易用**：支持 YAML 配置和简洁的命令行接口 (CLI)。
- ✅ **生产就绪**：提供完整的训练、推理和验证流水线。
- ✅ **多 GPU 支持**：支持通过 `--device` 扩展到多卡并行训练。

### 什么是 LoRA？

LoRA 不直接微调模型的所有权重，而是在原有权重旁边注入小的训练矩阵：
```
W' = W_frozen + B×A  (其中 rank << model_dim)
```
**结果**：仅需训练约 1% 的参数，原始模型权重保持冻结。

### 项目架构 (Architecture)

SAM3-LoRA 将低秩自适应应用于 SAM3 架构的核心组件：

<div align="center">
<img src="asset/Screenshot 2568-12-06 at 07.00.16.png" alt="SAM3 Architecture with LoRA" width="900">
<br>
<em>应用了完整 LoRA 适配的 SAM3 模型架构</em>
</div>

<br>

**LoRA 适配器应用于：**

| 组件 | 描述 | LoRA 影响 |
|-----------|-------------|-------------|
| **视觉编码器 (ViT)** | 从输入图像中提取视觉特征 | 高 - 核心特征学习 |
| **文本编码器** | 处理用于引导分割的文本提示 (Prompts) | 中 - 语义理解 |
| **几何编码器** | 处理几何提示（点、框） | 中 - 空间推理 |
| **DETR 编码器** | 用于目标检测的 Transformer 编码器 | 高 - 场景理解 |
| **DETR 解码器** | 用于对象查询的 Transformer 解码器 | 高 - 对象定位 |
| **掩码解码器 (Mask Decoder)** | 生成最终的分割掩码 | 高 - 精细化分割 |

---

## 安装指南 (Installation)

### 环境要求
- Python 3.8+, PyTorch 2.0+, CUDA (可选), 具有 SAM3 访问权限的 Hugging Face 账号。

### 安装步骤

1. **申请 SAM3 访问权限**：访问 [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3) 并提交申请。
2. **克隆并安装依赖**：
```bash
git clone https://github.com/doublefei0702/sam3_finetune_lora.git
cd sam3_finetune_lora
pip install -e .
```
3. **登录 Hugging Face**：
```bash
huggingface-cli login
# 输入你的 Access Token
```

---

## 快速入门 (Quick Start)

### 1. 准备数据
将数据集整理为 **COCO 格式** 每个拆分（train/valid）包含一个标注文件 or **Referring segmentation 格式**：

```
**COCO 格式** 
data/
├── train/                    # 必须
│   ├── img001.jpg
│   └── _annotations.coco.json
├── valid/                    # 强烈建议
│   ├── img001.jpg
│   └── _annotations.coco.json

**Referring segmentation 格式**
data/
├── images/
│   ├── train_1.jpg
│   ├── val_1.jpg
│   ├── test_1.jpg
├── binary_masks/      
│   ├── train_1.jpg   # 这里因为我是从语义分割数据集上构造的 referring masks，所谓我增加了一个层类别文件夹
│   ├── val_1.jpg
│   ├── test_1.jpg
```

### 2. 训练模型
使用默认配置启动训练：
```bash
# 单 GPU 自定义配置文件
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 scripts/training/train_sam3_lora_native.py --config configs/vaihingen_lora_config.yaml --device 0

# 多 GPU 自定义配置文件
torchrun --nproc_per_node 2 scripts/training/train_sam3_lora_native.py --config configs/vaihingen_lora_config.yaml --device 0 1
```
Training script computes validation loss only.

### 3. 执行推理
使用训练好的模型进行预测：

计算指标
```bash
python scripts/inference/validate_sam3_lora.py \
--config configs/vaihingen_lora_config.yaml \
--weights outputs/sam3_lora_vaihingen/best_lora_weights.pt \ 
--dataset-type vaihingen \ 
--vai-data-dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef \ 
--vai-split valid \
--vai-variant concept\
```

可视化
```bash
python scripts/inference/infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image test_image.jpg \
  --prompt "yellow school bus" \
  --output predictions.png
```


SAM3-LoRA uses a two-stage validation approach following SAM3's original design:
- During Training: Only validation loss is computed (fast, no expensive metrics)
- After Training: Run full evaluation with mAP, cgF1 metrics and NMS filtering

This approach significantly speeds up training while still monitoring overfitting via validation loss.


---


### 自定义配置 (YAML)
在配置文件中调整 LoRA 参数：
```yaml
lora:
  rank: 16                    # LoRA 秩 (越高容量越大)
  alpha: 32                   # 缩放因子 (通常为 2×rank)
  target_modules: ["q_proj", "v_proj"] # 目标层
  apply_to_vision_encoder: true        # 应用于视觉编码器
  apply_to_mask_decoder: true          # 应用于掩码解码器
```

---

## 推理使用 (Inference)

`infer_sam.py` 支持多种高级功能：

- **多文本提示**：`--prompt "crack" "damage"`
- **NMS 过滤**：通过 `--nms-iou 0.3` 减少重叠框（0.3 更严格，0.7 更宽松）。
- **边界框显示**：`--boundingbox True`
- **置信度阈值**：`--threshold 0.3`

---

## 常见问题排查 (Troubleshooting)

- **显存溢出 (OOM)**：减小 `batch_size` 或 `rank`。建议使用 `configs/light_lora_config.yaml`。
- **损失不下降**：尝试增大学习率 (`learning_rate`) 或 `rank`。
- **可训练参数过多**：如果超过 50%，说明基础模型未正确冻结（已在最新版修复）。
- **无验证数据**：脚本会运行但无法保存“最佳”模型，仅能保存“最后”一轮的模型。

---

## 项目结构 (Project Structure)

- `docs/`: 所有的项目指南和说明文档。
- `scripts/training/`: 训练脚本（单卡/多卡）。
- `scripts/inference/`: 推理、验证和模型对比工具。
- `scripts/tools/`: 数据预处理、分析工具。
- `sam3/`: SAM3 核心库。
- `lora_layers.py`: LoRA 核心实现逻辑。
- `ruff.toml`: 代码规范配置文件。

---

## Thanks

- [SAM3_LoRA](https://github.com/Sompote/SAM3_LoRA)
- [sam3_finetune_lora](https://github.com/doublefei0702/sam3_finetune_lora)

---

## 开源协议 (License)

本项目采用 Apache 2.0 协议。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**版本**: 1.0.0 | **构建者**: ICCLab, Leo

[⬆ 返回顶部](#sam3-lora-基于低秩自适应-lora-的高效微调)

</div>
