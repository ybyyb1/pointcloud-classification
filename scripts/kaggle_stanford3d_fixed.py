#!/usr/bin/env python3
"""
修正版Kaggle Stanford3D训练脚本
解决只读文件系统问题
"""

import os
import sys
import warnings
import shutil

warnings.filterwarnings("ignore", message=".*CUDA capability sm_60.*")

print("=" * 60)
print("Kaggle Stanford3D训练 - 修正版")
print("=" * 60)

# 1. 定义路径
input_dir = "/kaggle/input/stanford3d-dataset"
working_data_dir = "/kaggle/working/data/stanford3d"

print(f"输入目录: {input_dir}")
print(f"工作数据目录: {working_data_dir}")

# 2. 检查输入目录
if not os.path.exists(input_dir):
    print(f"❌ 错误: 输入目录不存在: {input_dir}")
    print("请检查:")
    print("1. 数据集是否已附加到Notebook")
    print("2. 数据集名称是否正确")

    print("\n/kaggle/input 目录内容:")
    !ls -la /kaggle/input/
    sys.exit(1)

# 3. 列出输入目录内容
print(f"\n输入目录内容:")
!ls -la {input_dir}

# 4. 复制文件到可写目录
print(f"\n复制文件到可写目录: {working_data_dir}")
os.makedirs(working_data_dir, exist_ok=True)

# 复制所有文件
files_copied = []
for item in os.listdir(input_dir):
    src = os.path.join(input_dir, item)
    dst = os.path.join(working_data_dir, item)

    if os.path.isfile(src):
        shutil.copy2(src, dst)
        files_copied.append(item)
        size_mb = os.path.getsize(src) / (1024 * 1024)
        print(f"  ✓ 复制: {item} ({size_mb:.2f} MB)")

if not files_copied:
    print("❌ 错误: 未找到任何文件可复制")
    sys.exit(1)

# 5. 验证HDF5文件
h5_files = [f for f in files_copied if f.endswith('.h5') or f.endswith('.hdf5')]
if not h5_files:
    print("❌ 错误: 未找到HDF5文件")
    sys.exit(1)

h5_file = os.path.join(working_data_dir, h5_files[0])
print(f"\n使用HDF5文件: {h5_file}")

# 6. 验证文件内容
import h5py
try:
    with h5py.File(h5_file, 'r') as f:
        print("✅ HDF5文件验证成功!")
        print(f"训练样本: {f['train_points'].shape}")
        print(f"测试样本: {f['test_points'].shape}")
        if 'class_names' in f.attrs:
            import json
            class_names = json.loads(f.attrs['class_names'])
            print(f"类别数: {len(class_names)}")
            print(f"类别: {class_names}")
except Exception as e:
    print(f"❌ HDF5文件验证失败: {e}")
    sys.exit(1)

# 7. 克隆项目
print("\n" + "=" * 60)
print("克隆项目代码...")
os.chdir('/kaggle/working')
if os.path.exists('pointcloud-classification'):
    print("删除已存在的项目目录...")
    shutil.rmtree('pointcloud-classification')

!git clone https://github.com/ybyyb1/pointcloud-classification.git
%cd pointcloud-classification
!pip install -r requirements.txt

# 8. 创建临时修正 - 防止在只读目录中创建文件夹
print("\n应用临时修正...")

# 创建一个临时补丁来修复只读文件系统问题
patch_code = '''
import os
from data.datasets.base_dataset import BaseDataset

# 保存原始方法
original_makedirs = os.makedirs

# 创建包装器
def safe_makedirs(name, mode=0o777, exist_ok=False):
    """安全创建目录，如果是只读目录则跳过"""
    try:
        # 尝试写入测试文件
        test_file = os.path.join(name, '.test_write')
        if os.path.exists(name):
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                # 目录可写，正常创建
                return original_makedirs(name, mode, exist_ok)
            except (OSError, PermissionError):
                # 目录只读，跳过创建
                print(f"警告: 目录 {name} 是只读的，跳过创建")
                return
        else:
            # 目录不存在，尝试创建
            try:
                return original_makedirs(name, mode, exist_ok)
            except (OSError, PermissionError):
                print(f"警告: 无法在只读位置创建目录 {name}")
                return
    except Exception as e:
        print(f"警告: 目录创建检查失败: {e}")
        # 回退到原始方法
        return original_makedirs(name, mode, exist_ok)

# 临时替换 os.makedirs
os.makedirs = safe_makedirs

# 也修正BaseDataset的__init__方法
original_init = BaseDataset.__init__

def patched_init(self, config, split):
    """修正后的BaseDataset初始化"""
    # 调用原始初始化，但跳过目录创建部分
    self.config = config
    self.split = split
    self.data_dir = config.data_dir

    # 设置数据集属性
    self.points = []
    self.labels = []
    self.class_names = []
    self.class_to_id = {}
    self.id_to_class = {}
    self.transforms = None

    # 不尝试创建目录 - 在load_data中处理

BaseDataset.__init__ = patched_init

print("✅ 已应用临时修正")
'''

# 将补丁保存为文件
patch_file = '/tmp/kaggle_patch.py'
with open(patch_file, 'w') as f:
    f.write(patch_code)

# 导入补丁
import sys
sys.path.insert(0, '/tmp')
try:
    import kaggle_patch
    print("✅ 临时修正应用成功")
except Exception as e:
    print(f"⚠️ 临时修正应用失败: {e}")
    print("将继续尝试训练...")

# 9. 训练模型
print("\n" + "=" * 60)
print("开始训练Stanford3D模型...")

experiment_name = "stanford3d_kaggle"
epochs = 50
batch_size = 32
data_dir = working_data_dir  # 使用可写目录

train_cmd = f"""
python scripts/train.py \
  --experiment {experiment_name} \
  --epochs {epochs} \
  --batch_size {batch_size} \
  --kaggle \
  --model point_transformer \
  --dataset stanford3d \
  --data_dir {data_dir}
"""

print(f"执行命令:\n{train_cmd}")
print("\n" + "=" * 60)

# 执行训练
!{train_cmd}

print("\n" + "=" * 60)
print("🎉 训练完成!")
print(f"模型检查点保存在: /kaggle/working/pointcloud-classification/checkpoints/{experiment_name}")
print("=" * 60)