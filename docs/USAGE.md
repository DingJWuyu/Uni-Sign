# Uni-Sign Usage Guide / Uni-Sign 使用指南

[English](#english) | [中文](#中文)

---

## English

### 📋 Table of Contents
- [System Requirements](#system-requirements)
- [Data Preparation Checklist](#data-preparation-checklist)
- [Usage Methods for Different Datasets](#usage-methods-for-different-datasets)
- [Inference Modes](#inference-modes)
- [Memory Optimization Tips](#memory-optimization-tips)
- [Common Issues & Troubleshooting](#common-issues--troubleshooting)

---

### 💻 System Requirements

#### Minimum Hardware Requirements

**For Training:**
- **GPU**: NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000, A6000)
  - Multi-GPU setup recommended: 4x GPUs for optimal performance
  - CUDA 12.1 or higher required
- **CPU**: 8+ cores recommended
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ free space (datasets are large)
  - CSL-News RGB: ~200GB
  - How2Sign: ~300GB
  - OpenASL: ~250GB

**For Inference (Pose-only, Lightweight):**
- **GPU**: NVIDIA GPU with at least 8GB VRAM (e.g., RTX 3060, RTX 4060)
- **CPU**: 4+ cores
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ (for model checkpoints and pose data only)

**For Inference (RGB-pose):**
- **GPU**: NVIDIA GPU with at least 16GB VRAM (e.g., RTX 3080, RTX 4070 Ti)
- **CPU**: 8+ cores
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ (for model checkpoints and processed videos; full datasets require more space)

#### Software Requirements
- Python 3.9
- PyTorch 2.1.1 with CUDA 12.1
- DeepSpeed 0.16.3 (for distributed training)
- See `requirements.txt` for complete dependencies

---

### 📦 Data Preparation Checklist

Before running the model, ensure you have prepared all necessary components:

#### 1. Pre-trained Weights
- [ ] Download [mt5-base](https://huggingface.co/google/mt5-base) weights
- [ ] Place in `./pretrained_weight/mt5-base` directory

#### 2. Dataset Files

**Choose based on your task:**

##### For CSL-News (Chinese Sign Language Translation):
- [ ] Download RGB videos from [HuggingFace](https://huggingface.co/datasets/ZechengLi19/CSL-News) or [BaiDu](https://pan.baidu.com/s/17W6kIreNMHYtD4y2llKmDg?pwd=ncvo)
- [ ] Download pose data from [HuggingFace](https://huggingface.co/datasets/ZechengLi19/CSL-News_pose)
- [ ] Label file: `./data/CSL_News/CSL_News_Labels.json` (included in repo)
- [ ] Extract to: `./dataset/CSL_News/`

##### For CSL-Daily (Chinese Sign Language Translation):
- [ ] Download from [CSL-Daily website](https://ustc-slr.github.io/datasets/2021_csl_daily/)
- [ ] Download pose data from [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign)
- [ ] Ensure `sentence-crop` folder exists (see [Issue #7](https://github.com/ZechengLi19/Uni-Sign/issues/7) if missing)
- [ ] Label files: `./data/CSL_Daily/labels.{train,dev,test}` (included in repo)
- [ ] Extract to: `./dataset/CSL_Daily/`

##### For WLASL (American Sign Language Recognition):
- [ ] Download from [WLASL GitHub](https://github.com/dxli94/WLASL)
- [ ] Download pose data from [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign)
- [ ] Label files: `./data/WLASL/labels-2000.{train,dev,test}` (included in repo)
- [ ] Extract to: `./dataset/WLASL/`

##### For How2Sign (American Sign Language Translation):
- [ ] Download from [How2Sign website](https://how2sign.github.io/)
- [ ] Download pose data from [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign)
- [ ] Merge split files: `cat how2sign_pose_format.zip.* > how2sign_pose_format.zip && unzip how2sign_pose_format.zip`
- [ ] Label files: `./data/How2Sign/labels.{train,test}` (included in repo)
- [ ] Extract to: `./dataset/How2Sign/`

##### For OpenASL (American Sign Language Translation):
- [ ] Download from [OpenASL GitHub](https://github.com/chevalierNoir/OpenASL)
- [ ] Download pose data from [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign)
- [ ] Merge split files: `cat openasl_pose_format.zip.* > openasl_pose_format.zip && unzip openasl_pose_format.zip`
- [ ] Label files: `./data/OpenASL/labels.{train,dev,test}` (included in repo)
- [ ] Extract to: `./dataset/OpenASL/`

#### 3. Model Checkpoints (for inference/fine-tuning)
- [ ] Download pre-trained checkpoints from [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign)
  - Stage 1 (pose-only): For lightweight inference
  - Stage 2 (RGB-pose): For full-featured inference
  - Stage 3 (fine-tuned): For specific downstream tasks

#### 4. Verify Configuration
- [ ] Update `config.py` if your data paths differ from defaults
- [ ] Ensure all paths in `config.py` match your actual file locations

---

### 🎯 Usage Methods for Different Datasets

#### Training Pipeline

**Stage 1: Pose-only Pre-training** (Most Lightweight)
```bash
# Recommended for: Limited GPU memory or pose-only applications
# Memory requirement: ~20GB per GPU with batch size 16
output_dir=out/stage1_pretraining

deepspeed --include localhost:0,1,2,3 --master_port 29511 pre_training.py \
   --batch-size 16 \
   --gradient-accumulation-steps 8 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --dataset CSL_News  # or CSL_Daily, WLASL, How2Sign, OpenASL
```

**Stage 2: RGB-pose Pre-training**
```bash
# Recommended for: Full performance, requires more memory
# Memory requirement: ~24GB per GPU with batch size 4
output_dir=out/stage2_pretraining
ckpt_path=out/stage1_pretraining/best_checkpoint.pth

deepspeed --include localhost:0,1,2,3 --master_port 29511 pre_training.py \
   --batch-size 4 \
   --gradient-accumulation-steps 8 \
   --epochs 5 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset CSL_News \
   --rgb_support
```

**Stage 3: Downstream Fine-tuning**

*For Sign Language Translation (SLT):*
```bash
output_dir=out/stage3_finetuning
ckpt_path=out/stage2_pretraining/best_checkpoint.pth  # or stage1 for pose-only

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset CSL_Daily \
  --task SLT \
  --rgb_support
  
# Alternative datasets: CSL_News, How2Sign, OpenASL
# For pose-only mode, remove the --rgb_support flag
```

*For Isolated Sign Language Recognition (ISLR):*
```bash
output_dir=out/stage3_finetuning
ckpt_path=out/stage2_pretraining/best_checkpoint.pth

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset WLASL \
  --task ISLR \
  --max_length 64 \
  --rgb_support  # remove for pose-only
```

#### Evaluation

```bash
# Evaluate on single GPU
bash ./script/eval_stage3.sh
```

---

### 🚀 Inference Modes

Uni-Sign supports two inference modes with different performance and resource requirements:

#### Mode 1: Pose-only Inference (Recommended for Lightweight Applications)

**Advantages:**
- Lower GPU memory requirement (8GB+)
- Faster inference speed (~2-3x faster)
- Smaller model size
- Suitable for real-time applications

**Use Cases:**
- Resource-constrained environments
- Real-time sign language recognition
- Mobile or edge deployment

**Command:**
```bash
ckpt_path=out/stage3_finetuning/best_checkpoint.pth

python ./demo/online_inference.py \
   --online_video {video_path} \
   --finetune $ckpt_path
```

#### Mode 2: RGB-pose Inference (Best Accuracy)

**Advantages:**
- Higher accuracy
- Better performance on complex signs
- Utilizes both visual and pose information

**Use Cases:**
- Offline processing
- High-accuracy requirements
- Research and benchmarking

**Command:**
```bash
ckpt_path=out/stage3_finetuning/best_checkpoint.pth

python ./demo/online_inference.py \
   --online_video {video_path} \
   --finetune $ckpt_path \
   --rgb_support
```

#### Pose Extraction from Videos

If you have raw videos and need to extract pose data:

```bash
# Install dependencies first
cd ./demo/rtmlib-main
pip install -e .
cd ../../

# Extract poses
python ./demo/pose_extraction.py \
    --src_dir {video_dir} \
    --tgt_dir {pose_dir}
```

---

### ⚡ Memory Optimization Tips

#### For Training:

1. **Reduce Batch Size:**
   - Decrease `--batch-size` and increase `--gradient-accumulation-steps` to maintain effective batch size
   - Example: `--batch-size 4 --gradient-accumulation-steps 16` instead of `--batch-size 16 --gradient-accumulation-steps 4`

2. **Use Pose-only Mode:**
   - Remove `--rgb_support` flag to train with pose data only
   - Reduces memory usage by ~40% compared to RGB-pose mode

3. **Gradient Checkpointing:**
   - Enabled by default in the model
   - Trades computation for memory

4. **Mixed Precision Training:**
   - DeepSpeed automatically uses FP16/BF16
   - Configured in DeepSpeed config files

5. **Reduce Sequence Length:**
   - Decrease `--max_length` parameter for shorter sequences
   - Particularly useful for ISLR tasks

#### For Inference:

1. **Use Pose-only Checkpoints:**
   - Load Stage 1 checkpoints instead of Stage 2/3
   - Requires only pose data, no RGB frames

2. **Batch Size:**
   - Process videos one at a time for minimal memory usage
   - Increase batch size if memory allows for faster processing

3. **Video Resolution:**
   - Reduce input video resolution before pose extraction
   - 256x256 or 512x512 is often sufficient

4. **Model Quantization (Advanced):**
   - Consider INT8 quantization for production deployment
   - Requires additional post-training steps

---

### 🔧 Common Issues & Troubleshooting

#### Issue 1: Out of Memory (OOM) Errors

**Symptoms:** `CUDA out of memory` error during training/inference

**Solutions:**
- Reduce batch size: `--batch-size 2` or `--batch-size 1`
- Use pose-only mode (remove `--rgb_support`)
- Use fewer GPUs with larger gradient accumulation
- Clear GPU cache: Add `torch.cuda.empty_cache()` calls if modifying code

#### Issue 2: Missing Dataset Files

**Symptoms:** `FileNotFoundError` when starting training

**Solutions:**
- Verify all paths in `config.py` match your actual file locations
- Check that label files exist in `./data/{dataset}/`
- Ensure RGB/pose folders exist in `./dataset/{dataset}/`
- For CSL-Daily, verify `sentence-crop` folder exists (see [Issue #7](https://github.com/ZechengLi19/Uni-Sign/issues/7))

#### Issue 3: DeepSpeed Configuration

**Symptoms:** DeepSpeed fails to initialize or hangs

**Solutions:**
- Ensure all GPUs specified in `--include` are available
- Check CUDA and NCCL are properly installed
- Verify network ports (e.g., 29511) are not in use
- For single GPU: `deepspeed --include localhost:0 ...`

#### Issue 4: Slow Pose Extraction

**Symptoms:** Pose extraction takes very long

**Solutions:**
- Ensure `onnxruntime-gpu` is installed (not CPU version)
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Process videos in batches
- Use lower video resolution

#### Issue 5: mt5-base Download Issues

**Symptoms:** Cannot download or load mt5-base weights

**Solutions:**
- Use mirror sites if HuggingFace is blocked
- Download manually and place in `./pretrained_weight/mt5-base/`
- Verify all required files are present: `config.json`, `tokenizer.json`, `pytorch_model.bin`, etc.

#### Issue 6: Checkpoint Loading Errors

**Symptoms:** `RuntimeError: Error(s) in loading state_dict`

**Solutions:**
- Ensure checkpoint matches the model configuration (pose-only vs RGB-pose)
- Don't mix Stage 1 checkpoints with RGB-pose models
- Check PyTorch version compatibility

---

### 📚 Additional Resources

- **Main README**: [README.md](../README.md)
- **Dataset Preparation**: [DATASET.md](./DATASET.md)
- **Demo and Inference**: [demo/README.md](../demo/README.md)
- **Pre-training Code**: [Issue #15](https://github.com/ZechengLi19/Uni-Sign/issues/15)
- **Model Checkpoints**: [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign)

---

### 📮 Support

If you encounter issues not covered in this guide, please:
1. Check existing [GitHub Issues](https://github.com/ZechengLi19/Uni-Sign/issues)
2. Open a new issue with detailed error messages and system information
3. Contact: Zecheng Li (lizecheng19@gmail.com)

---

## 中文

### 📋 目录
- [系统要求](#系统要求)
- [数据准备清单](#数据准备清单)
- [不同数据集的使用方法](#不同数据集的使用方法)
- [推理模式](#推理模式)
- [内存优化技巧](#内存优化技巧)
- [常见问题与故障排除](#常见问题与故障排除)

---

### 💻 系统要求

#### 最低硬件要求

**训练场景：**
- **GPU**：至少 24GB 显存的 NVIDIA GPU（如 RTX 3090、RTX 4090、A5000、A6000）
  - 推荐多GPU配置：4个GPU以获得最佳性能
  - 需要 CUDA 12.1 或更高版本
- **CPU**：推荐 8 核以上
- **内存**：64GB+ 系统内存
- **存储空间**：500GB+ 可用空间（数据集较大）
  - CSL-News RGB：约 200GB
  - How2Sign：约 300GB
  - OpenASL：约 250GB

**推理场景（仅姿态，轻量化）：**
- **GPU**：至少 8GB 显存的 NVIDIA GPU（如 RTX 3060、RTX 4060）
- **CPU**：4 核以上
- **内存**：16GB+ 系统内存
- **存储空间**：50GB+（仅用于模型检查点和姿态数据）

**推理场景（RGB-姿态）：**
- **GPU**：至少 16GB 显存的 NVIDIA GPU（如 RTX 3080、RTX 4070 Ti）
- **CPU**：8 核以上
- **内存**：32GB+ 系统内存
- **存储空间**：100GB+

#### 软件要求
- Python 3.9
- PyTorch 2.1.1 with CUDA 12.1
- DeepSpeed 0.16.3（用于分布式训练）
- 完整依赖项请参见 `requirements.txt`

---

### 📦 数据准备清单

在运行模型之前，请确保已准备好所有必要组件：

#### 1. 预训练权重
- [ ] 下载 [mt5-base](https://huggingface.co/google/mt5-base) 权重
- [ ] 放置在 `./pretrained_weight/mt5-base` 目录中

#### 2. 数据集文件

**根据您的任务选择：**

##### CSL-News（中国手语翻译）：
- [ ] 从 [HuggingFace](https://huggingface.co/datasets/ZechengLi19/CSL-News) 或 [百度云](https://pan.baidu.com/s/17W6kIreNMHYtD4y2llKmDg?pwd=ncvo) 下载 RGB 视频
- [ ] 从 [HuggingFace](https://huggingface.co/datasets/ZechengLi19/CSL-News_pose) 下载姿态数据
- [ ] 标签文件：`./data/CSL_News/CSL_News_Labels.json`（仓库中已包含）
- [ ] 解压到：`./dataset/CSL_News/`

##### CSL-Daily（中国手语翻译）：
- [ ] 从 [CSL-Daily 网站](https://ustc-slr.github.io/datasets/2021_csl_daily/) 下载
- [ ] 从 [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign) 下载姿态数据
- [ ] 确保 `sentence-crop` 文件夹存在（如缺失请参见 [Issue #7](https://github.com/ZechengLi19/Uni-Sign/issues/7)）
- [ ] 标签文件：`./data/CSL_Daily/labels.{train,dev,test}`（仓库中已包含）
- [ ] 解压到：`./dataset/CSL_Daily/`

##### WLASL（美国手语识别）：
- [ ] 从 [WLASL GitHub](https://github.com/dxli94/WLASL) 下载
- [ ] 从 [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign) 下载姿态数据
- [ ] 标签文件：`./data/WLASL/labels-2000.{train,dev,test}`（仓库中已包含）
- [ ] 解压到：`./dataset/WLASL/`

##### How2Sign（美国手语翻译）：
- [ ] 从 [How2Sign 网站](https://how2sign.github.io/) 下载
- [ ] 从 [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign) 下载姿态数据
- [ ] 合并分割文件：`cat how2sign_pose_format.zip.* > how2sign_pose_format.zip && unzip how2sign_pose_format.zip`
- [ ] 标签文件：`./data/How2Sign/labels.{train,test}`（仓库中已包含）
- [ ] 解压到：`./dataset/How2Sign/`

##### OpenASL（美国手语翻译）：
- [ ] 从 [OpenASL GitHub](https://github.com/chevalierNoir/OpenASL) 下载
- [ ] 从 [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign) 下载姿态数据
- [ ] 合并分割文件：`cat openasl_pose_format.zip.* > openasl_pose_format.zip && unzip openasl_pose_format.zip`
- [ ] 标签文件：`./data/OpenASL/labels.{train,dev,test}`（仓库中已包含）
- [ ] 解压到：`./dataset/OpenASL/`

#### 3. 模型检查点（用于推理/微调）
- [ ] 从 [HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign) 下载预训练检查点
  - Stage 1（仅姿态）：用于轻量化推理
  - Stage 2（RGB-姿态）：用于全功能推理
  - Stage 3（微调后）：用于特定下游任务

#### 4. 验证配置
- [ ] 如果数据路径与默认值不同，请更新 `config.py`
- [ ] 确保 `config.py` 中的所有路径与实际文件位置匹配

---

### 🎯 不同数据集的使用方法

#### 训练流程

**Stage 1：仅姿态预训练**（最轻量化）
```bash
# 推荐用于：GPU显存有限或仅姿态应用
# 显存需求：batch size 16 时每个GPU约 20GB
output_dir=out/stage1_pretraining

deepspeed --include localhost:0,1,2,3 --master_port 29511 pre_training.py \
   --batch-size 16 \
   --gradient-accumulation-steps 8 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --dataset CSL_News  # 或 CSL_Daily, WLASL, How2Sign, OpenASL
```

**Stage 2：RGB-姿态预训练**
```bash
# 推荐用于：完整性能，需要更多显存
# 显存需求：batch size 4 时每个GPU约 24GB
output_dir=out/stage2_pretraining
ckpt_path=out/stage1_pretraining/best_checkpoint.pth

deepspeed --include localhost:0,1,2,3 --master_port 29511 pre_training.py \
   --batch-size 4 \
   --gradient-accumulation-steps 8 \
   --epochs 5 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset CSL_News \
   --rgb_support
```

**Stage 3：下游任务微调**

*手语翻译 (SLT)：*
```bash
output_dir=out/stage3_finetuning
ckpt_path=out/stage2_pretraining/best_checkpoint.pth  # 或使用 stage1 进行仅姿态训练

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset CSL_Daily \
  --task SLT \
  --rgb_support
  
# 其他数据集选项：CSL_News, How2Sign, OpenASL
# 要使用仅姿态模式，请移除 --rgb_support 标志
```

*孤立手语识别 (ISLR)：*
```bash
output_dir=out/stage3_finetuning
ckpt_path=out/stage2_pretraining/best_checkpoint.pth

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset WLASL \
  --task ISLR \
  --max_length 64 \
  --rgb_support  # 移除此项以使用仅姿态模式
```

#### 评估

```bash
# 在单个GPU上评估
bash ./script/eval_stage3.sh
```

---

### 🚀 推理模式

Uni-Sign 支持两种推理模式，具有不同的性能和资源要求：

#### 模式 1：仅姿态推理（推荐用于轻量化应用）

**优势：**
- 更低的GPU显存需求（8GB+）
- 更快的推理速度（约快 2-3 倍）
- 更小的模型大小
- 适合实时应用

**使用场景：**
- 资源受限环境
- 实时手语识别
- 移动端或边缘设备部署

**命令：**
```bash
ckpt_path=out/stage3_finetuning/best_checkpoint.pth

python ./demo/online_inference.py \
   --online_video {video_path} \
   --finetune $ckpt_path
```

#### 模式 2：RGB-姿态推理（最佳准确度）

**优势：**
- 更高的准确度
- 在复杂手势上表现更好
- 同时利用视觉和姿态信息

**使用场景：**
- 离线处理
- 高精度要求
- 研究和基准测试

**命令：**
```bash
ckpt_path=out/stage3_finetuning/best_checkpoint.pth

python ./demo/online_inference.py \
   --online_video {video_path} \
   --finetune $ckpt_path \
   --rgb_support
```

#### 从视频提取姿态

如果您有原始视频并需要提取姿态数据：

```bash
# 首先安装依赖
cd ./demo/rtmlib-main
pip install -e .
cd ../../

# 提取姿态
python ./demo/pose_extraction.py \
    --src_dir {video_dir} \
    --tgt_dir {pose_dir}
```

---

### ⚡ 内存优化技巧

#### 训练优化：

1. **减少批量大小：**
   - 降低 `--batch-size` 并增加 `--gradient-accumulation-steps` 以保持有效批量大小
   - 示例：使用 `--batch-size 4 --gradient-accumulation-steps 16` 代替 `--batch-size 16 --gradient-accumulation-steps 4`

2. **使用仅姿态模式：**
   - 移除 `--rgb_support` 标志以仅使用姿态数据训练
   - 相比 RGB-姿态模式减少约 40% 的显存使用

3. **梯度检查点：**
   - 模型中默认启用
   - 用计算换取显存

4. **混合精度训练：**
   - DeepSpeed 自动使用 FP16/BF16
   - 在 DeepSpeed 配置文件中配置

5. **减少序列长度：**
   - 降低 `--max_length` 参数以处理更短序列
   - 对 ISLR 任务特别有用

#### 推理优化：

1. **使用仅姿态检查点：**
   - 加载 Stage 1 检查点而非 Stage 2/3
   - 仅需要姿态数据，无需 RGB 帧

2. **批量大小：**
   - 一次处理一个视频以最小化显存使用
   - 如果显存允许，可增加批量大小以加快处理速度

3. **视频分辨率：**
   - 在姿态提取前降低输入视频分辨率
   - 256x256 或 512x512 通常足够

4. **模型量化（高级）：**
   - 考虑 INT8 量化用于生产部署
   - 需要额外的训练后步骤

---

### 🔧 常见问题与故障排除

#### 问题 1：显存不足 (OOM) 错误

**症状：** 训练/推理期间出现 `CUDA out of memory` 错误

**解决方案：**
- 减小批量大小：`--batch-size 2` 或 `--batch-size 1`
- 使用仅姿态模式（移除 `--rgb_support`）
- 使用更少的GPU配合更大的梯度累积
- 清除GPU缓存：如果修改代码，添加 `torch.cuda.empty_cache()` 调用

#### 问题 2：数据集文件缺失

**症状：** 启动训练时出现 `FileNotFoundError`

**解决方案：**
- 验证 `config.py` 中的所有路径与实际文件位置匹配
- 检查标签文件是否存在于 `./data/{dataset}/`
- 确保 RGB/姿态文件夹存在于 `./dataset/{dataset}/`
- 对于 CSL-Daily，验证 `sentence-crop` 文件夹存在（参见 [Issue #7](https://github.com/ZechengLi19/Uni-Sign/issues/7)）

#### 问题 3：DeepSpeed 配置

**症状：** DeepSpeed 初始化失败或挂起

**解决方案：**
- 确保 `--include` 中指定的所有GPU都可用
- 检查 CUDA 和 NCCL 是否正确安装
- 验证网络端口（如 29511）未被占用
- 对于单GPU：使用 `deepspeed --include localhost:0 ...`

#### 问题 4：姿态提取速度慢

**症状：** 姿态提取耗时很长

**解决方案：**
- 确保安装了 `onnxruntime-gpu`（而非CPU版本）
- 验证 CUDA 可用：`python -c "import torch; print(torch.cuda.is_available())"`
- 批量处理视频
- 使用较低的视频分辨率

#### 问题 5：mt5-base 下载问题

**症状：** 无法下载或加载 mt5-base 权重

**解决方案：**
- 如果 HuggingFace 被屏蔽，使用镜像站点
- 手动下载并放置在 `./pretrained_weight/mt5-base/`
- 验证所有必需文件都存在：`config.json`、`tokenizer.json`、`pytorch_model.bin` 等

#### 问题 6：检查点加载错误

**症状：** `RuntimeError: Error(s) in loading state_dict`

**解决方案：**
- 确保检查点与模型配置匹配（仅姿态 vs RGB-姿态）
- 不要将 Stage 1 检查点与 RGB-姿态模型混用
- 检查 PyTorch 版本兼容性

---

### 📚 其他资源

- **主README**：[README.md](../README.md)
- **数据集准备**：[DATASET.md](./DATASET.md)
- **演示和推理**：[demo/README.md](../demo/README.md)
- **预训练代码**：[Issue #15](https://github.com/ZechengLi19/Uni-Sign/issues/15)
- **模型检查点**：[HuggingFace](https://huggingface.co/ZechengLi19/Uni-Sign)

---

### 📮 技术支持

如果遇到本指南未涵盖的问题，请：
1. 查看现有的 [GitHub Issues](https://github.com/ZechengLi19/Uni-Sign/issues)
2. 提交包含详细错误信息和系统信息的新 issue
3. 联系：Zecheng Li (lizecheng19@gmail.com)
