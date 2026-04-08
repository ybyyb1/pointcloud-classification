#!/usr/bin/env python3
"""
Kaggle Stanford3D数据集设置脚本
解决只读文件系统问题
"""

import os
import sys
import shutil
import glob

def setup_stanford3d_for_kaggle():
    """设置Stanford3D数据集在Kaggle环境中使用"""

    print("=" * 60)
    print("Kaggle Stanford3D数据集设置")
    print("=" * 60)

    # 1. 定义路径
    input_dir = "/kaggle/input/stanford3d-dataset"
    working_dir = "/kaggle/working/data/stanford3d"
    processed_dir = os.path.join(working_dir, "processed")

    print(f"输入目录: {input_dir}")
    print(f"工作目录: {working_dir}")
    print(f"处理目录: {processed_dir}")

    # 2. 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        print("\n可用的输入目录:")
        !ls -la /kaggle/input/
        return False

    # 3. 列出输入文件
    print(f"\n输入目录内容:")
    input_files = os.listdir(input_dir)
    for f in input_files:
        file_path = os.path.join(input_dir, f)
        size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  {f}: {size:.2f} MB")

    # 4. 创建目标目录结构
    print(f"\n创建目录结构...")
    os.makedirs(processed_dir, exist_ok=True)

    # 5. 复制文件
    print(f"\n复制文件...")
    files_copied = []

    # 复制所有文件到processed目录
    for filename in input_files:
        src = os.path.join(input_dir, filename)
        dst = os.path.join(processed_dir, filename)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
            files_copied.append(filename)
            print(f"  ✓ {filename}")

    if not files_copied:
        print("❌ 错误: 未复制任何文件")
        return False

    # 6. 验证HDF5文件
    h5_files = [f for f in files_copied if f.endswith(('.h5', '.hdf5'))]
    if not h5_files:
        print("❌ 错误: 未找到HDF5文件")
        return False

    h5_file = os.path.join(processed_dir, h5_files[0])
    print(f"\nHDF5文件: {h5_file}")

    # 7. 验证文件
    import h5py
    try:
        with h5py.File(h5_file, 'r') as f:
            print("✅ HDF5文件验证成功!")
            print(f"训练样本: {f['train_points'].shape}")
            print(f"测试样本: {f['test_points'].shape}")

            # 获取类别信息
            if 'class_names' in f.attrs:
                import json
                class_names = json.loads(f.attrs['class_names'])
                print(f"类别数: {len(class_names)}")
                print(f"类别: {class_names[:5]}...")  # 只显示前5个
            else:
                print("⚠️ 警告: 未找到类别信息")

    except Exception as e:
        print(f"❌ HDF5文件验证失败: {e}")
        return False

    # 8. 创建符号链接（可选）
    print(f"\n创建必要的符号链接...")

    # 检查是否需要创建原始数据目录的符号链接
    raw_data_dir = os.path.join(working_dir, "Stanford3dDataset_v1.2")
    if not os.path.exists(raw_data_dir):
        # 创建空目录，防止数据集类尝试下载
        os.makedirs(raw_data_dir, exist_ok=True)
        print(f"创建空目录: {raw_data_dir}")

    # 9. 总结
    print("\n" + "=" * 60)
    print("设置完成!")
    print(f"数据目录: {working_dir}")
    print(f"HDF5文件: {h5_file}")
    print(f"文件总数: {len(files_copied)}")
    print("\n训练命令:")
    print(f"python scripts/train.py \\")
    print(f"  --experiment stanford3d_kaggle \\")
    print(f"  --epochs 50 \\")
    print(f"  --batch_size 32 \\")
    print(f"  --kaggle \\")
    print(f"  --model point_transformer \\")
    print(f"  --dataset stanford3d \\")
    print(f"  --data_dir {working_dir}")
    print("=" * 60)

    return True


def create_kaggle_notebook_cell():
    """创建Kaggle Notebook单元格代码"""

    code = '''# Kaggle Stanford3D训练单元格
import os
import shutil

# 1. 设置路径
input_dir = "/kaggle/input/stanford3d-dataset"
working_dir = "/kaggle/working/data/stanford3d"
processed_dir = os.path.join(working_dir, "processed")

print("设置Stanford3D数据集...")

# 2. 创建目录结构
os.makedirs(processed_dir, exist_ok=True)

# 3. 复制文件
for filename in os.listdir(input_dir):
    src = os.path.join(input_dir, filename)
    dst = os.path.join(processed_dir, filename)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"复制: {filename}")

# 4. 验证
import h5py
h5_files = [f for f in os.listdir(processed_dir) if f.endswith('.h5')]
if h5_files:
    h5_file = os.path.join(processed_dir, h5_files[0])
    with h5py.File(h5_file, 'r') as f:
        print(f"训练样本: {f['train_points'].shape}")
        print(f"测试样本: {f['test_points'].shape}")
        print("✅ 数据集准备完成!")

# 5. 克隆项目
os.chdir('/kaggle/working')
!git clone https://github.com/ybyyb1/pointcloud-classification.git
%cd pointcloud-classification
!pip install -r requirements.txt

# 6. 训练
!python scripts/train.py \\
  --experiment stanford3d_kaggle \\
  --epochs 50 \\
  --batch_size 32 \\
  --kaggle \\
  --model point_transformer \\
  --dataset stanford3d \\
  --data_dir {working_dir}
'''

    return code


if __name__ == "__main__":
    # 如果直接运行，执行设置
    success = setup_stanford3d_for_kaggle()

    if success:
        print("\n" + "=" * 60)
        print("复制以下代码到Kaggle Notebook:")
        print("=" * 60)
        print(create_kaggle_notebook_cell())
        sys.exit(0)
    else:
        sys.exit(1)