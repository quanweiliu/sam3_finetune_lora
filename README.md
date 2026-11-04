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
  - sam3Trainer 从 train_sam3_lora_datav2 中拆分出的训练文件
  - config
    - potsdam_lora_config.ymal
    - vaihingen_lora_config.ymal
    - 记得修改 output_dir, 每次运行都要修改。
  - scripts
    - data
      - isprs_dataset_mix_ss.py
      - isprs_dataset_mix.py
      - isprs_dataset_ss.py
      - isprs_dataset.py
    - inference
      - infer_vis.py    默认加载 SAM3 权重推理，也可加载自己训练的权重，一次只能推理一张图像，通常做单张图像的可视化，or 体验 SAM3 的效果。
      - isprs_sam3_lora.py    默认加载 SAM3 权重推理，也可加载自己训练的权重，在 val folder 上推理，可以得出在验证集上的精度。
      - isprs_sam3_lora_label0.py   Pesudo-label 生成：只是简单的生成伪标签+过滤置信度较低的标签（160-163 行）。
      - isprs_sam3_lora_label1.py   Pesudo-label 生成：逐类别生成不同的权重
    - train
      - train_sam3_lora_native.py   模型训练文件
      - train_sam3_lora_native_ss.py   模型训练文件
      - train_sam3_lora_native_mix.py   模型训练文件
      - train_sam3_lora_native_mix_ss.py   模型训练文件
      - train_sam3_lora_datav2    v2模型训练文件
      - train_sam3_lora_datav2_simplify    v2模型训练文件, 拆分出 SAM3TrainerNative
    - 模型支持只有 mask 的 concept prompt segmentation, 通过 mask 直接推理 bbox

- 拆分出数据和一些辅助函数：
  - 拆分以下几个文件
    - lqw/sam3_finetune_lora/scripts/training/train_sam3_lora_native.py
    - lqw/sam3_finetune_lora/scripts/inference/validate_sam3_lora.py
    - lqw/sam3_finetune_lora/scripts/inference/infer_sam.py
  - 数据加载部分
    - 拆分到 lqw/sam3_finetune_lora/scripts/data/COCODataset.py and lqw/sam3_finetune_lora/scripts/data/vaihingen_dataset.py
  - 辅助函数部分
    - 拆分到 /home/icclab/Documents/lqw/sam3_finetune_lora/scripts/tools/utils.py


- 表述：
  - train_simple.py Small Transformer created inside the script; Minimal LoRA **demonstration**
  - train_standalone.py (line 27)	Small Transformer created inside the script; Configurable LoRA demonstration with a reusable trainer
  - train_sam3_lora_native Native SAM3 via build_sam3_image_model; Custom SAM3 LoRA fine-tuning with single- or **multi-GPU** support


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

## VPRef v0.7 全类别45px忽略门（当前入口）

规则：`../VPRefDocu/VPRef_v07_dataset_rules.md`；数据：`../DatasetMMF/VPRef_Vaihingen_v07/`。
同一52张图，先忽略五类中原生mask面积<45px的连通区（恰好45保留），再重新执行全部v06家族、
排除、英文多样性和每图最多5条抽样。原始标注、v06及其复核记录保持。
忽略区不参与目标／地标／排名／分组／排除；`ignore_masks/`和每条`ignore_mask_path`记录其像素。

继续复用`VPRefV03Dataset`、`dataset: vpref_v03`与`target_mode: union`。更新后的训练脚本
在mask focal/Dice中排除ignore像素；评估的像素指标也排除它们。忽略图通过独立collator字段传给损失，
不会输入模型；图像不遮盖，框／分类损失仍针对保留目标。旧版本无ignore字段时沿用原损失与指标。
复核页绿色为目标、橙色为ignore；原语义图及区域ID图保留以便追溯。

```bash
python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v07_lora_config.yaml --check-data
```

配置仍支持`training.task_families`及验证继承，不需要导出额外子集。输出独立到`outputs/vpref_v07/`。
检查报告见数据包`metadata/report.md`与`metadata/validation.json`；新查询待人工复核，val仍为开发数据。
本轮只构造和验证数据，没有启动训练。

## VPRef v0.6 排除与每图上限

规则：`../VPRefDocu/VPRef_v06_dataset_rules.md`；数据：`../DatasetMMF/VPRef_Vaihingen_v06/`。
沿用52张图及全部v05候选规则，新增`ref.exclusion`，每图从合格主查询中固定种子42随机保留最多5条。
最终37张train共185条、15张val共75条，其中排除查询28/14条。
少于5条合格候选时全部保留；旋转标签不额外占名额。没有按家族/类别强制配额；
如组内中间未抽中，`metadata/rule_coverage.json`明确标记`qualified_not_sampled`。
完整候选池与抽样决定分别在`metadata/candidate_pool.jsonl`、`metadata/sampling.jsonl`。
它们是审计材料，不是额外训练清单。新排除表达仍待人工复核，val仍作开发。

```bash
python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v06_lora_config.yaml --check-data
```

继续复用`VPRefV03Dataset`和`target_mode: union`，没有新增数据加载器，也没有启动训练。
训练配置输出独立到`outputs/vpref_v06/`，旧数据、权重和结果保持。

直接按家族训练，无需导出新的JSONL。在同一配置中设置：

```yaml
training:
  manifest: train.jsonl
  task_families: [ref.exclusion]
  validation:
    manifest: val.jsonl
    allow_unreviewed_proposals: true
output:
  output_dir: outputs/vpref_v06_exclusion
```

保留其他配置项。验证集默认继承`training.task_families`，因此本例选择28条train和14条val。
`task_families: []`、`null`或省略表示全部；多个家族可写为`[ref.exclusion, ref.ordinal]`。
若验证集要单独选择，在`training.validation.task_families`中设置；显式`[]`或`null`可重置为全部。
筛选只读取现有manifest，不改数据或补回随机抽样未选中的家族；不存在的家族名会报错并列出可用家族。
原先的`train_exclusion.jsonl`/`val_exclusion.jsonl`仍可使用，但无需为每次实验新增清单。

v06英文已扩展为`vpref-equivalent-english-1.0`受控同义表述。每个家族都注册语言池，
仅替换原查询的主英文并同步C4方向；260条查询、ID、mask与每图上限均保持。
数据包`metadata/language_diversity.json`报告各家族改写前后不同英文数，
`index.html`和`reviews.csv`显示改写前英文供对照，新措辞仍需人工语言复核。
原始英文及工件保存在数据包`metadata/language_previous.zip`，旧v05不修改。

评估默认沿用保存的训练配置，也可只在评估时覆盖家族：

```bash
python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v06_exclusion --split val --task-families ref.exclusion
```

`--all-task-families`可覆盖配置筛选、评估原manifest中的全部家族。筛选参数、实际家族数量和选中ID
会记录在训练选择文件、数据检查报告及评估protocol/summary中。

## VPRef v0.5 统一规则数据

统一构造规范：`../VPRefDocu/VPRef_v05_dataset_rules.md`；数据位于
`../DatasetMMF/VPRef_Vaihingen_v05/`。沿用原 52 张图及 train/val 场景划分，
恢复中央附加描述、组内中间、序数、车辆长度与大小、轮廓极值、最近关系和依赖式推理。
五类全集和 v04 四向集合保留；只有对角措辞关闭，不删四向集合里的对角位置目标。

当前数据量及逐家族覆盖见数据包 `metadata/summary.json`、`metadata/rule_coverage.json`；
零合格样例有明确拒绝原因。`index.html` 和 `metadata/reviews.csv` 用于人工复核，
程序通过的新样例仍是待审提案。val 仍属开发数据。

```bash
python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v05_lora_config.yaml --check-data
```

继续使用 `VPRefV03Dataset`、`dataset: vpref_v03`、`target_mode: union`；
`dataset_version=0.5` 表示内容版本。配置独立输出到 `outputs/vpref_v05/`。
这一步只构造、验证数据，没有启动训练。旧版数据、配置和权重保留。

## VPRef v0.4 五类数据

数据目录：`../DatasetMMF/VPRef_Vaihingen_v04/`，37 张 train / 15 张 val，
284 条 train / 112 条 val 查询。五类都有类别全集；另有合格单区域和汽车四向地标查询。
只删除四种对角表达，四向答案和中心线 ±10px 筛选保持；不因车辆处于对角位置而从四向答案删车。
数据目录的 `index.html` 可点击表达检查目标，`metadata/validation.json` 为独立检查结果。
构造依据见 `../VPRefDocu/VPRef_v04_dataset_rules.md`。

```bash
python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v04_lora_config.yaml --check-data
```

v0.4 复用 `VPRefV03Dataset` 和已有 union-mask schema，故配置中的 backend 名称仍为
`dataset: vpref_v03`；内容版本为 `dataset_version=0.4`。不需要再复制一套加载器。
配置的训练输出为 `outputs/vpref_v04/`。本轮仅构造和检查数据，没有启动训练。

## VPRef v0.3 小规模试验

当前学习／调试入口为独立数据目录
`../DatasetMMF/VPRef_Vaihingen_v03/`，包含 83 train / 30 val 查询及其自身的 images、masks。
使用 `configs/vpref_v03_centerline_filtered_lora_config.yaml`；该配置已指向新目录，
输出到 `outputs/vpref_v03_debug/`，不会覆盖已有实验权重。
数据文件及规则和原筛选子集一致，仅整理路径；`metadata/` 保存来源和几何依据。
下面的小包、扩展批次与历史输出说明用于追溯原实验。

Vaihingen 首轮使用 8 条已审查询、5 张影像，480 × 480、union-mask 监督。
训练配置为 `configs/vpref_v03_lora_config.yaml`；当前保存的权重是
`outputs/sam3_lora_vpref_v03_vaihingen_union_480/last_lora_weights.pt`。

```bash
# 数据与 SAM3 组批检查（不加载模型权重）
python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v03_lora_config.yaml --check-data

# 训练集拟合评估；结果包含逐查询 IoU、Dice、mask 和 index.html
python scripts/inference/isprs_sam3_lora_datav2.py --device 0

# 固定当前权重：base 与 LoRA 各推理一次，缓存 float32 logits 后统一比较五种后处理
python scripts/inference/compare_vpref_v03_postprocessing.py --device 0
```

对照条件为 union@0.3/0.5/0.7、top1@0.3、top1@0，mask 阈值统一为 0.5。
top1@0 是全为正样本时的诊断，不作为空查询的部署策略。未按每条查询的真值选阈值。
对照结果位于训练输出目录的 `postprocessing_baseline_comparison/`；
没有可用 CUDA 时只生成计划和 waiting_for_cuda 状态，不产生新的模型对照数值。
完成后查看 `comparison.csv`、`report.md` 及每个条件下的 `index.html`。
所有当前评估均为训练集诊断，不能当作独立测试性能。

新的待复核扩展批次位于
`../DatasetMMF/VPRef_v2_expression_pilot/v03_vaihingen_expansion_01/`：
37 张 train / 138 条查询，15 张原始 val / 55 条开发验证查询。
先通过统一 `index.html` 与 `reviews.csv`、`region_reviews.csv` 完成人工复核，
再导出正式训练／开发验证清单。

按用户要求，另提供扩展提案的实验训练配置（标签仍待复核）：

```bash
# 原始 train：138 条查询 / 37 张图；原始 val：55 条查询 / 15 张图
python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v03_expansion_lora_config.yaml --check-data
python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v03_expansion_lora_config.yaml --device 0
```

`training.validation.manifest` 指向 `experimental_training/val.jsonl`，
`require_validation: true` 会在验证清单缺失、为空或与训练原场景重叠时停止。
验证固定使用原始方向，关闭梯度和由真值采样的交互提示；每个 epoch 计算验证 loss，
写入 `val_stats.json`（JSONL），并保存最低验证 loss 的 `best_lora_weights.pt`。
选中的验证查询和来源分别保存在 `selected_val_samples.jsonl`、`validation_data.json`。
输出目录为 `outputs/sam3_lora_vpref_v03_expansion_unreviewed_vaihingen_union_480_with_val/`。
这批 val 用于开发和模型选择，标签尚未人工接受，不代表独立金标准测试。
原 8 条已审小包配置 `vpref_v03_lora_config.yaml` 继续保持训练集拟合诊断用途，不自动添加验证集。

同一个推理脚本支持显式选择 train／val，例如检查之前的一步 smoke 权重：

```bash
python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v03_validation_smoke --split val --check-data
python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v03_validation_smoke --split val --device 0
python scripts/inference/isprs_sam3_lora_datav2.py --run-dir outputs/vpref_v03_validation_smoke --split train --device 0
```

默认仍为 `--split train`，默认权重为保存目录内的 `last_lora_weights.pt`。
val 从保存配置的 `training.validation` 读取清单，结果保存到 `vpref_v03_val_lora/`；
train 保存到 `vpref_v03_fit_lora/`。每个目录包含逐查询指标、预测 mask、参考 mask、
四联预览和 `index.html`。阈值默认 score=0.3、mask=0.5，选择方式为 union。
`vpref_v03_validation_smoke` 只进行过一个 optimizer step，适合检查推理流程，
不可将其指标视为完整训练结果。未复核参考标签和训练／验证来源均在报告中明确标注。

新增建筑中心线查询筛选（原目标与英文不变）：水平／竖直线穿过可见建筑 mask 质心，
两侧各 10 px 内出现任一面积 >=20 px 的车辆轮廓时，分别跳过上下／左右表达。
对角表达任一所需方向被阻断就跳过。现有扩展 train 138→83（32 图）、val 55→30（14 图）。
对照页及跳过原因见原批次下的 `centerline_filtered/index.html`；原 193 条查询和历史评估保留。

```bash
python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v03_centerline_filtered_lora_config.yaml --check-data
# 如需训练该筛选子集：使用独立输出目录，83 train + 30 原 val 开发查询
python scripts/training/train_sam3_lora_datav2.py --config configs/vpref_v03_centerline_filtered_lora_config.yaml --device 0
```

此配置保持原模型／优化器参数；当前 train/val manifest 已整理到上述独立数据目录，
使用单独的 debug 输出目录。
筛选条件是在查看 val 后提出的；子集上的分数不能直接与原 55 条 val 的指标解释为模型提升。

`isprs_vis_datav2.py` 可直接把已保存的预测 mask 叠加到输入影像，无需重新推理：

```bash
python scripts/inference/isprs_vis_datav2.py \
  --results-dir outputs/sam3_lora_vpref_v03_vaihingen_union_480_centerline10_with_val/vpref_v03_val_lora \
  --overlay --overlay-alpha 0.45
```

`--overlay` 默认开启；`--no-overlay` 改为黑底彩色 mask；`--overlay-alpha` 控制叠加透明度（0–1），
`--overlay-color "255,60,60"` 或 `--overlay-color "#ff3c3c"` 控制颜色，
`--no-masks` 仅显示原图。输出默认为结果目录内的 `visualization/`，包含原尺寸 PNG、
原图副本及 `index.html`；可用 `--output` 指定目录。重复运行会另存，不覆盖已有图。
可视化复用原始预测，不修改 mask、阈值或指标。
原单图推理模式继续支持 `--file_path RUN --image IMAGE --prompt "cars"` 及相同叠加参数，
用 `--device 0` 选择 GPU；该模式沿用自身的置信度／NMS 设置，查看已评估结果时应使用 `--results-dir`。

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
