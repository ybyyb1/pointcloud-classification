# Kaggle 集成指南

本指南介绍如何在Kaggle上使用点云分类系统。

## 先决条件

1. **Kaggle 账户**: 需要Kaggle账户
2. **GitHub 仓库**: 项目应在GitHub上 (https://github.com/ybyyb1/pointcloud-classification)
3. **Kaggle Notebook**: 在Kaggle上创建新notebook (https://www.kaggle.com/code)

## 快速开始（推荐）

在Kaggle Notebook中运行以下代码块：

```python
# 单元格1: 克隆仓库和设置
!git clone https://github.com/ybyyb1/pointcloud-classification.git
import os
os.chdir('/kaggle/working/pointcloud-classification')
!pip install -r requirements.txt

# 单元格2: 设置Kaggle API密钥（使用你的凭证）
import os
os.environ['KAGGLE_USERNAME'] = 'ybyyb1'
os.environ['KAGGLE_KEY'] = 'KGAT_e15b251e1961531282207d55cc009ceb'

# 或者使用脚本设置
!python scripts/setup_kaggle_api.py

# 单元格3: 验证系统
!python scripts/verify_system.py

# 单元格4: 下载数据集（自动使用Kaggle数据集）
!python main.py download-scanobjectnn --data_dir /kaggle/working/data/scanobjectnn

# 单元格4: 训练模型
!python scripts/train.py \
  --experiment kaggle_training \
  --epochs 50 \
  --batch_size 16 \
  --kaggle \
  --model point_transformer \
  --dataset scanobjectnn \
  --data_dir /kaggle/working/data/scanobjectnn
```

如果遇到数据集下载问题，请确保已接受Kaggle数据集的使用条款：
- 访问 https://www.kaggle.com/datasets/hkustvgd/scanobjectnn
- 点击 "New Notebook" 或 "Accept Rules"

## 选项1: 自动设置（详细）

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

以下是Kaggle notebook的完整示例，包含错误处理和验证：

```python
# 单元格1: 设置和验证
import os
import sys
import warnings

# 抑制CUDA兼容性警告
warnings.filterwarnings("ignore", message=".*CUDA capability sm_60.*")

# 克隆仓库
!git clone https://github.com/ybyyb1/pointcloud-classification.git
os.chdir('/kaggle/working/pointcloud-classification')

# 安装依赖
!pip install -r requirements.txt

# 设置Kaggle API密钥（使用你的凭证）
print("设置Kaggle API密钥...")
os.environ['KAGGLE_USERNAME'] = 'ybyyb1'
os.environ['KAGGLE_KEY'] = 'KGAT_e15b251e1961531282207d55cc009ceb'

# 或者使用脚本设置
!python scripts/setup_kaggle_api.py

# 验证系统
print("验证系统...")
!python scripts/verify_system.py

# 单元格2: 数据集准备
print("\n准备数据集...")

# 下载数据集（自动使用Kaggle数据集）
print("使用你的Kaggle API密钥下载数据集...")
!python main.py download-scanobjectnn --data_dir /kaggle/working/data/scanobjectnn

# 单元格3: 训练模型
print("\n开始训练...")

# 训练配置
experiment_name = "kaggle_training"
epochs = 50  # Kaggle限制，建议30-50轮
batch_size = 16  # P100 GPU适合的批次大小

train_cmd = f"""
python scripts/train.py \
  --experiment {experiment_name} \
  --epochs {epochs} \
  --batch_size {batch_size} \
  --kaggle \
  --model point_transformer \
  --dataset scanobjectnn \
  --data_dir /kaggle/working/data/scanobjectnn
"""

print(f"执行命令:\n{train_cmd}")
!{train_cmd}

# 单元格4: 评估和可视化
print("\n评估模型...")

# 查找最新的检查点
checkpoint_dir = f"/kaggle/working/pointcloud-classification/checkpoints/{experiment_name}"
if os.path.exists(checkpoint_dir):
    import glob
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"使用检查点: {latest_checkpoint}")
        
        # 评估
        !python main.py evaluate \
          --checkpoint {latest_checkpoint} \
          --dataset scanobjectnn \
          --data_dir /kaggle/working/data/scanobjectnn
        
        # 可视化
        !python main.py visualize \
          --dataset scanobjectnn \
          --num_samples 5 \
          --output_dir /kaggle/working/visualizations
    else:
        print("未找到检查点文件")
else:
    print(f"检查点目录不存在: {checkpoint_dir}")

print("\n🎉 Kaggle训练流程完成！")
print(f"模型检查点保存在: {checkpoint_dir}")
print(f"可视化结果保存在: /kaggle/working/visualizations")
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

### 5. **快速设置Kaggle API**
项目提供了专门的脚本设置Kaggle API密钥：

```python
# 方法1: 使用预配置的用户 (ybyyb1)
!python scripts/setup_kaggle_api.py

# 方法2: 手动设置
!python scripts/setup_kaggle_api.py --username ybyyb1 --api-key KGAT_e15b251e1961531282207d55cc009ceb --token-name bishe

# 方法3: 直接在代码中设置环境变量
import os
os.environ['KAGGLE_USERNAME'] = 'ybyyb1'
os.environ['KAGGLE_KEY'] = 'KGAT_e15b251e1961531282207d55cc009ceb'
```

**你的凭证信息:**
- 用户名: `ybyyb1`
- API密钥: `KGAT_e15b251e1961531282207d55cc009ceb`
- 令牌名称: `bishe`

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

### 问题: 验证脚本失败 "Can't instantiate abstract class BaseModel"
**解决方案**: 此问题已在最新版本中修复。确保使用最新的代码：
```python
# 重新克隆仓库获取最新修复
!rm -rf /kaggle/working/pointcloud-classification
!git clone https://github.com/ybyyb1/pointcloud-classification.git
```

或手动修复验证脚本：
```python
# 编辑 scripts/verify_system.py，将 "创建虚拟模型" 测试改为使用具体模型
```

### 问题: CUDA兼容性警告 (Tesla P100-PCIE-16GB with CUDA capability sm_60)
**说明**: 这是警告信息，不影响训练。Kaggle的P100 GPU计算能力为6.0，而PyTorch 2.10+需要7.0+，但PyTorch会回退到CPU模式运行兼容的核函数。训练仍可正常进行。

**解决方案**: 忽略此警告，或使用以下代码抑制警告：
```python
import warnings
warnings.filterwarnings("ignore", message=".*CUDA capability sm_60.*")
```

### 问题: Kaggle API认证失败
**解决方案**: 设置Kaggle API密钥
```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_key'
```
或使用Kaggle Secrets：
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ['KAGGLE_USERNAME'] = secrets.get_secret("KAGGLE_USERNAME")
os.environ['KAGGLE_KEY'] = secrets.get_secret("KAGGLE_KEY")
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