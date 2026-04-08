# 快速开始指南

本指南将帮助您快速安装和运行点云分类系统。

## ⏱️ 5分钟快速体验

### 1. 环境准备

确保您的系统满足以下要求：
- **Python**: 3.8 或更高版本
- **PyTorch**: 2.0 或更高版本
- **GPU** (可选但推荐): NVIDIA GPU, CUDA 11.0+

### 2. 快速安装

```bash
# 克隆项目
git clone https://github.com/yourusername/pointcloud-classification-system.git
cd pointcloud-classification-system

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装核心依赖
pip install torch torchvision numpy
pip install -r requirements.txt
```

### 3. 验证安装

```bash
# 检查PyTorch是否安装正确
python -c "import torch; print(f'PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}')"

# 检查项目模块
python -c "from models.point_transformer import PointTransformer; print('PointTransformer导入成功')"
```

## 🚀 完整安装流程

### 步骤1: 系统要求检查

#### Windows 用户
```powershell
# 检查Python版本
python --version

# 检查pip版本
pip --version

# 如果未安装Python，请从官网下载: https://www.python.org/downloads/
```

#### Linux/Mac 用户
```bash
# 检查Python版本
python3 --version

# 安装虚拟环境工具 (如果需要)
sudo apt-get install python3-venv  # Ubuntu/Debian
# 或
brew install python3               # Mac
```

### 步骤2: 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 验证虚拟环境
which python  # Linux/Mac 应该显示 venv/bin/python
# 或
where python  # Windows 应该显示 venv\Scripts\python.exe
```

### 步骤3: 安装依赖

#### 基本依赖
```bash
# 升级pip
pip install --upgrade pip

# 安装PyTorch (根据您的CUDA版本选择)
# 无GPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 更多版本请查看: https://pytorch.org/get-started/locally/
```

#### 项目依赖
```bash
# 安装项目依赖
pip install -r requirements.txt

# 如果安装较慢，可以使用清华镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤4: 验证安装

创建测试脚本 `test_installation.py`:

```python
#!/usr/bin/env python3
"""
安装验证脚本
"""

import sys
print(f"Python版本: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch导入失败: {e}")
    sys.exit(1)

# 测试项目模块
modules_to_test = [
    ("models.point_transformer", "PointTransformer"),
    ("data.datasets.scanobjectnn_dataset", "ScanObjectNNDataset"),
    ("training.trainer", "Trainer"),
    ("config.base_config", "SystemConfig"),
]

for module_name, class_name in modules_to_test:
    try:
        exec(f"from {module_name} import {class_name}")
        print(f"✓ {module_name}.{class_name} 导入成功")
    except ImportError as e:
        print(f"✗ {module_name}.{class_name} 导入失败: {e}")

print("\n✅ 安装验证完成!")
```

运行验证脚本:
```bash
python test_installation.py
```

## 🎯 第一个示例

### 示例1: 下载数据集

```bash
# 创建数据目录
mkdir -p data/scanobjectnn

# 下载ScanObjectNN数据集 (小样本版本用于测试)
python main.py download-scanobjectnn \
  --data_dir ./data/scanobjectnn \
  --version main_split \
  --num_points 1024
```

### 示例2: 快速训练

```bash
# 使用小规模配置进行测试训练
python scripts/train.py \
  --experiment test_run \
  --epochs 5 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --model point_transformer \
  --dataset scanobjectnn \
  --data_dir ./data/scanobjectnn
```

### 示例3: 使用预训练配置

```bash
# 创建示例配置
mkdir -p configs
python main.py create-config \
  --project_name "快速测试" \
  --model point_transformer \
  --dataset scanobjectnn \
  --output configs/quick_test.yaml

# 使用配置训练
python main.py train --config configs/quick_test.yaml
```

## 💡 常见问题

### Q1: 安装过程中遇到权限问题
```bash
# 不要使用sudo安装Python包
# 正确做法是使用虚拟环境
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Q2: PyTorch安装失败
- 确认Python版本 (需要3.8+)
- 确认pip已更新 (`pip install --upgrade pip`)
- 查看PyTorch官方安装指南: https://pytorch.org/get-started/locally/

### Q3: CUDA不可用
```python
import torch
print(torch.cuda.is_available())  # 应该返回True
```
如果返回False:
- 确认已安装NVIDIA驱动
- 确认CUDA版本与PyTorch版本匹配
- 可以使用CPU版本进行测试

### Q4: 导入模块失败
```bash
# 确保在项目根目录运行
pwd  # 应该显示 pointcloud-classification-system

# 确保虚拟环境已激活
which python  # 应该显示 venv/bin/python
```

## 🚀 下一步

完成安装后，您可以:
1. 查看 [命令行界面指南](cli_guide.md) 了解所有可用命令
2. 查看 [数据集指南](datasets.md) 了解如何准备数据
3. 查看 [训练指南](training.md) 了解完整训练流程
4. 运行 [Jupyter笔记本](../notebooks/pointcloud_classification_demo.ipynb) 进行交互式学习

## ⚠️ 注意事项

1. **数据集下载**: ScanObjectNN数据集约1GB，S3DIS数据集约70GB，请确保有足够磁盘空间
2. **训练时间**: 完整训练可能需要数小时到数天，建议使用GPU
3. **内存要求**: 训练时需要4GB+系统内存，建议8GB以上
4. **首次运行**: 首次运行时会下载预训练权重和数据集，请保持网络连接

---

**恭喜您完成安装!** 🎉 现在可以开始使用点云分类系统了。