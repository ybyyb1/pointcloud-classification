# Kaggle 集成指南

本指南介绍如何在Kaggle上使用点云分类系统。

## 先决条件

1. **Kaggle 账户**: 需要Kaggle账户
2. **GitHub 仓库**: 项目应在GitHub上 (https://github.com/ybyyb1/pointcloud-classification)
3. **Kaggle Notebook**: 在Kaggle上创建新notebook (https://www.kaggle.com/code)

## 选项1: 自动设置（推荐）

### 步骤1: 在Kaggle Notebook中克隆和设置

在Kaggle notebook的第一个单元格中添加以下代码：

```python
# 克隆GitHub仓库
!git clone https://github.com/ybyyb1/pointcloud-classification.git

# 切换到项目目录
import os
os.chdir('/kaggle/working/pointcloud-classification')

# 安装依赖
!pip install -r requirements.txt

# 设置Kaggle环境
!python scripts/setup_kaggle_github.py
```

### 步骤2: 下载数据集

```python
# 下载ScanObjectNN数据集
!python main.py download-scanobjectnn --data_dir /kaggle/working/data/scanobjectnn
```

### 步骤3: 使用Kaggle优化训练模型

```python
# 使用Kaggle优化训练
!python scripts/train.py \
  --experiment kaggle_training \
  --epochs 50 \
  --batch_size 16 \
  --kaggle \
  --data_dir /kaggle/working/data/scanobjectnn \
  --model point_transformer \
  --dataset scanobjectnn
```

### 步骤4: 评估模型

```python
# 评估训练好的模型
!python main.py evaluate \
  --checkpoint /kaggle/working/checkpoints/kaggle_training/best_model.pth \
  --dataset scanobjectnn \
  --data_dir /kaggle/working/data/scanobjectnn
```

## 选项2: 手动设置

### 步骤1: 克隆仓库

```python
import os
import sys

# 克隆仓库
!git clone https://github.com/ybyyb1/pointcloud-classification.git

# 添加到Python路径
project_path = '/kaggle/working/pointcloud-classification'
sys.path.insert(0, project_path)
os.chdir(project_path)
```

### 步骤2: 安装依赖

```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -r requirements.txt
```

### 步骤3: 导入项目模块

```python
# 导入项目模块
from main import download_scanobjectnn, train_model, evaluate_model
from config import SystemConfig
import argparse
```

### 步骤4: 准备数据集

```python
# 创建参数命名空间
class Args:
    def __init__(self):
        self.data_dir = '/kaggle/working/data/scanobjectnn'
        self.version = 'main_split'
        self.num_points = 1024
        self.batch_size = 16

# 下载数据集
args = Args()
download_scanobjectnn(args)
```

### 步骤5: 训练模型

```python
# 创建训练配置
config = SystemConfig()

# 为Kaggle更新配置
config.training.epochs = 50
config.training.batch_size = 16
config.training.use_amp = True
config.training.checkpoint_dir = '/kaggle/working/checkpoints'
config.dataset.data_dir = '/kaggle/working/data/scanobjectnn'

# 运行训练
from scripts.train import train_model as train_func
results = train_func(
    experiment_name='kaggle_manual',
    use_kaggle=True,
    num_epochs=50
)

print(f"最佳准确率: {results['best_accuracy']:.4f}")
```

## Kaggle特定优化

### GPU内存管理

```python
import torch

# 检查GPU内存
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU内存: {gpu_memory:.2f} GB")
    
    # 为Kaggle P100 (16GB) 设置内存比例
    if gpu_memory >= 16:
        torch.cuda.set_per_process_memory_fraction(14.0 / gpu_memory)
```

### 性能优化

```python
# 设置环境变量优化性能
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
```

### Kaggle数据集集成

如果想使用Kaggle数据集：

```python
# 挂载Kaggle数据集
import opendatasets as od
od.download("https://www.kaggle.com/datasets/hkustvgd/scanobjectnn")

# 使用数据集
!python main.py train \
  --data_dir /kaggle/input/scanobjectnn \
  --kaggle
```

## 完整的Kaggle Notebook示例

以下是Kaggle notebook的完整示例：

```python
# 单元格1: 设置
!git clone https://github.com/ybyyb1/pointcloud-classification.git
import os
os.chdir('/kaggle/working/pointcloud-classification')
!pip install -r requirements.txt

# 单元格2: 数据集
!python main.py download-scanobjectnn --data_dir /kaggle/working/data/scanobjectnn

# 单元格3: 训练
!python scripts/train.py \
  --experiment my_kaggle_experiment \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --kaggle \
  --model point_transformer \
  --dataset scanobjectnn

# 单元格4: 评估
!python main.py evaluate \
  --checkpoint /kaggle/working/checkpoints/my_kaggle_experiment/best_model.pth \
  --dataset scanobjectnn

# 单元格5: 可视化
!python main.py visualize \
  --dataset scanobjectnn \
  --num_samples 5 \
  --output_dir /kaggle/working/visualizations
```

## Kaggle竞赛提交

如果参加Kaggle竞赛：

```python
# 创建提交文件
import pandas as pd
import numpy as np

# 从模型加载预测
predictions = np.load('/kaggle/working/predictions.npy')

# 创建提交DataFrame
submission = pd.DataFrame({
    'id': range(len(predictions)),
    'prediction': predictions
})

# 保存提交文件
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("提交文件已创建")
```

## Kaggle成功技巧

### 1. **高效使用Kaggle GPU**
- 批次大小: P100建议16-32
- 启用混合精度训练 (`--kaggle` 标志自动启用)
- 使用梯度累积获得更大的有效批次大小

### 2. **管理磁盘空间**
- Kaggle提供约20GB磁盘空间
- 清理中间文件：
  ```python
  !rm -rf /kaggle/working/__pycache__
  !rm -rf /kaggle/working/*.pyc
  ```

### 3. **保存检查点**
- 将模型检查点保存到 `/kaggle/working/checkpoints/`
- Kaggle Notebook输出会自动保存

### 4. **使用Kaggle Secrets存储API密钥**
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()

# 获取API密钥（如果需要）
api_key = secrets.get_secret("kaggle_api_key")
```

## 故障排除

### 问题: 内存不足
**解决方案**: 减小批次大小
```python
# 使用更小的批次大小
!python scripts/train.py --batch_size 8 --kaggle
```

### 问题: 训练速度慢
**解决方案**: 启用cuDNN基准测试
```python
import torch
torch.backends.cudnn.benchmark = True
```

### 问题: 导入错误
**解决方案**: 检查Python路径
```python
import sys
print(sys.path)
sys.path.insert(0, '/kaggle/working/pointcloud-classification')
```

### 问题: 数据集下载失败
**解决方案**: 使用Kaggle数据集
```python
# 从Kaggle数据集下载
!kaggle datasets download -d hkustvgd/scanobjectnn
!unzip scanobjectnn.zip -d /kaggle/working/data/
```

## 资源

- [Kaggle Notebook文档](https://www.kaggle.com/docs/notebooks)
- [Kaggle GPU指南](https://www.kaggle.com/docs/gpu)
- [点云分类GitHub](https://github.com/ybyyb1/pointcloud-classification)

## 支持

如果遇到问题：
1. 查看[常见问题](faq.md)
2. 提交[GitHub Issues](https://github.com/ybyyb1/pointcloud-classification/issues)
3. 在[Kaggle讨论区](https://www.kaggle.com/discussions)提问