# Quick Start Guide

This guide will help you quickly install and run the point cloud classification system.

## ⏱️ 5-Minute Quick Experience

### 1. Environment Preparation

Ensure your system meets the following requirements:
- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher
- **GPU** (optional but recommended): NVIDIA GPU, CUDA 11.0+

### 2. Quick Installation

```bash
# Clone the project
git clone https://github.com/yourusername/pointcloud-classification-system.git
cd pointcloud-classification-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install core dependencies
pip install torch torchvision numpy
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check if PyTorch is installed correctly
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Check project modules
python -c "from models.point_transformer import PointTransformer; print('PointTransformer imported successfully')"
```

## 🚀 Complete Installation Process

### Step 1: System Requirements Check

#### Windows Users
```powershell
# Check Python version
python --version

# Check pip version
pip --version

# If Python is not installed, download from: https://www.python.org/downloads/
```

#### Linux/Mac Users
```bash
# Check Python version
python3 --version

# Install virtual environment tools (if needed)
sudo apt-get install python3-venv  # Ubuntu/Debian
# or
brew install python3               # Mac
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Verify virtual environment
which python  # Linux/Mac should show venv/bin/python
# or
where python  # Windows should show venv\Scripts\python.exe
```

### Step 3: Install Dependencies

#### Basic Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose based on your CUDA version)
# CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# More versions at: https://pytorch.org/get-started/locally/
```

#### Project Dependencies
```bash
# Install project dependencies
pip install -r requirements.txt

# If installation is slow, use Tsinghua mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Step 4: Verify Installation

Create test script `test_installation.py`:

```python
#!/usr/bin/env python3
"""
Installation verification script
"""

import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

# Test project modules
modules_to_test = [
    ("models.point_transformer", "PointTransformer"),
    ("data.datasets.scanobjectnn_dataset", "ScanObjectNNDataset"),
    ("training.trainer", "Trainer"),
    ("config.base_config", "SystemConfig"),
]

for module_name, class_name in modules_to_test:
    try:
        exec(f"from {module_name} import {class_name}")
        print(f"✓ {module_name}.{class_name} imported successfully")
    except ImportError as e:
        print(f"✗ {module_name}.{class_name} import failed: {e}")

print("\n✅ Installation verification completed!")
```

Run verification script:
```bash
python test_installation.py
```

## 🎯 First Examples

### Example 1: Download Dataset

```bash
# Create data directory
mkdir -p data/scanobjectnn

# Download ScanObjectNN dataset (small sample version for testing)
python main.py download-scanobjectnn \
  --data_dir ./data/scanobjectnn \
  --version main_split \
  --num_points 1024
```

### Example 2: Quick Training

```bash
# Test training with small-scale configuration
python scripts/train.py \
  --experiment test_run \
  --epochs 5 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --model point_transformer \
  --dataset scanobjectnn \
  --data_dir ./data/scanobjectnn
```

### Example 3: Use Pre-configured Settings

```bash
# Create example configuration
mkdir -p configs
python main.py create-config \
  --project_name "Quick Test" \
  --model point_transformer \
  --dataset scanobjectnn \
  --output configs/quick_test.yaml

# Train using configuration
python main.py train --config configs/quick_test.yaml
```

## 💡 Frequently Asked Questions

### Q1: Permission issues during installation
```bash
# Do not use sudo to install Python packages
# Correct approach is to use virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Q2: PyTorch installation failed
- Confirm Python version (needs 3.8+)
- Confirm pip is updated (`pip install --upgrade pip`)
- Check PyTorch official installation guide: https://pytorch.org/get-started/locally/

### Q3: CUDA not available
```python
import torch
print(torch.cuda.is_available())  # Should return True
```
If returns False:
- Confirm NVIDIA driver is installed
- Confirm CUDA version matches PyTorch version
- You can use CPU version for testing

### Q4: Module import failed
```bash
# Ensure running from project root directory
pwd  # Should show pointcloud-classification-system

# Ensure virtual environment is activated
which python  # Should show venv/bin/python
```

## 🚀 Next Steps

After installation, you can:
1. Check [Command Line Interface Guide](cli_guide.md) to learn all available commands
2. Check [Dataset Guide](datasets.md) to learn how to prepare data
3. Check [Training Guide](training.md) to learn complete training process
4. Run [Jupyter notebook](../notebooks/pointcloud_classification_demo.ipynb) for interactive learning

## ⚠️ Important Notes

1. **Dataset download**: ScanObjectNN dataset ~1GB, S3DIS dataset ~70GB, ensure sufficient disk space
2. **Training time**: Complete training may take hours to days, recommend using GPU
3. **Memory requirements**: 4GB+ system memory required for training, recommend 8GB+
4. **First run**: Pre-trained weights and datasets will be downloaded on first run, keep network connection

---

**Congratulations on completing installation!** 🎉 Now you can start using the point cloud classification system.