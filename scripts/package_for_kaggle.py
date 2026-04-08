#!/usr/bin/env python3
"""
打包Stanford3D预处理数据用于Kaggle上传

将预处理后的Stanford3D数据集打包成zip文件，方便上传到Kaggle。
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
import argparse


def package_stanford3d_for_kaggle(
    processed_dir: str = "./data/stanford3d/processed",
    output_zip: str = "./stanford3d_processed_kaggle.zip",
    include_npz: bool = False,
    create_readme: bool = True
) -> str:
    """
    打包Stanford3D预处理数据用于Kaggle上传

    Args:
        processed_dir: 预处理数据目录
        output_zip: 输出的zip文件路径
        include_npz: 是否包含NPZ文件（会增加文件大小）
        create_readme: 是否创建README文件

    Returns:
        str: 生成的zip文件路径
    """
    # 检查输入目录
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        raise FileNotFoundError(f"预处理目录不存在: {processed_dir}")

    # 检查关键文件
    required_files = ["stanford3d_dataset.h5", "metadata.json"]
    missing_files = []
    for file in required_files:
        if not (processed_path / file).exists():
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(f"缺少必要文件: {missing_files}")

    print("=" * 60)
    print("打包Stanford3D预处理数据用于Kaggle上传")
    print("=" * 60)
    print(f"预处理目录: {processed_path.absolute()}")
    print(f"输出文件: {output_zip}")

    # 创建临时目录
    temp_dir = Path("./temp_kaggle_package")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        # 复制主要文件
        print("\n复制主要文件...")
        files_to_copy = ["stanford3d_dataset.h5", "metadata.json"]
        for file in files_to_copy:
            src = processed_path / file
            dst = temp_dir / file
            shutil.copy2(src, dst)
            file_size = src.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file}: {file_size:.2f} MB")

        # 可选：复制NPZ文件
        if include_npz:
            print("\n复制NPZ文件...")
            npz_files = ["train_data.npz", "val_data.npz", "test_data.npz"]
            for file in npz_files:
                src = processed_path / file
                if src.exists():
                    dst = temp_dir / file
                    shutil.copy2(src, dst)
                    file_size = src.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {file}: {file_size:.2f} MB")

        # 创建README文件
        if create_readme:
            print("\n创建README文件...")
            readme_content = create_readme_file(processed_path)
            readme_path = temp_dir / "README.md"
            readme_path.write_text(readme_content, encoding='utf-8')
            print(f"  ✓ README.md 已创建")

        # 创建dataset-metadata.json（Kaggle需要）
        print("\n创建Kaggle数据集元数据...")
        metadata = create_kaggle_metadata(processed_path)
        metadata_path = temp_dir / "dataset-metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"  ✓ dataset-metadata.json 已创建")

        # 创建zip文件
        print(f"\n创建zip文件: {output_zip}")
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)
                    print(f"  ✓ 添加: {arcname}")

        # 显示统计信息
        zip_size = Path(output_zip).stat().st_size / (1024 * 1024)
        print(f"\n打包完成!")
        print(f"文件大小: {zip_size:.2f} MB")
        print(f"输出路径: {output_zip}")

        # 显示上传建议
        print("\n" + "=" * 60)
        print("上传到Kaggle的步骤:")
        print("1. 访问 https://www.kaggle.com/datasets")
        print("2. 点击 'New Dataset'")
        print("3. 上传此zip文件")
        print("4. 填写数据集信息:")
        print("   - 名称: stanford3d-processed")
        print("   - 描述: Stanford3D Dataset v1.2 Preprocessed for Object Classification")
        print("   - 类别: Computer Vision → Point Clouds")
        print("5. 发布数据集")
        print("=" * 60)

        return output_zip

    finally:
        # 清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def create_readme_file(processed_path: Path) -> str:
    """创建README文件"""
    # 读取元数据
    metadata_path = processed_path / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 读取HDF5文件信息
    import h5py
    h5_path = processed_path / "stanford3d_dataset.h5"
    with h5py.File(h5_path, 'r') as f:
        num_train = f.attrs.get('num_train_samples', 0)
        num_test = f.attrs.get('num_test_samples', 0)
        num_classes = f.attrs.get('num_classes', 0)
        class_names = json.loads(f.attrs.get('class_names', '[]'))

    readme = f"""# Stanford3D Preprocessed Dataset

这是一个从Stanford3D Dataset v1.2预处理得到的点云分类数据集，格式与ScanObjectNN兼容。

## 数据集信息

### 基本信息
- **总实例数**: {metadata.get('total_samples', 'N/A')}
- **训练样本**: {num_train}
- **测试样本**: {num_test}
- **类别数**: {num_classes}
- **每个点云点数**: 1024

### 类别信息
数据集包含以下{len(class_names)}个类别：
{chr(10).join(f'- {cls}' for cls in class_names)}

### 类别分布
```json
{json.dumps(metadata.get('class_distribution', {}), indent=2, ensure_ascii=False)}
```

### 处理区域
- **已处理区域**: {metadata.get('areas_processed', [])}
- **分割比例**: {json.dumps(metadata.get('split_ratios', {}), indent=2, ensure_ascii=False)}

## 文件说明

### 主要文件
1. **stanford3d_dataset.h5** - HDF5格式数据集（与ScanObjectNN兼容）
   - `train_points`: 训练集点云 (形状: [{num_train}, 1024, 3])
   - `train_labels`: 训练集标签 (形状: [{num_train}, 1])
   - `test_points`: 测试集点云 (形状: [{num_test}, 1024, 3])
   - `test_labels`: 测试集标签 (形状: [{num_test}, 1])

2. **metadata.json** - 数据集元数据信息

### 可选文件
3. **train_data.npz, val_data.npz, test_data.npz** - 原始NPZ格式数据

## 使用方法

### Python加载
```python
import h5py
import numpy as np

# 加载HDF5文件
with h5py.File('stanford3d_dataset.h5', 'r') as f:
    train_points = f['train_points'][:]  # (N_train, 1024, 3)
    train_labels = f['train_labels'][:].flatten()  # (N_train,)
    test_points = f['test_points'][:]  # (N_test, 1024, 3)
    test_labels = f['test_labels'][:].flatten()  # (N_test,)

    # 获取类别信息
    class_names = json.loads(f.attrs['class_names'])
    num_classes = f.attrs['num_classes']
```

### 在点云分类项目中使用
```bash
# 下载数据集后
python scripts/train.py \\
  --experiment stanford3d_kaggle \\
  --epochs 50 \\
  --batch_size 32 \\
  --kaggle \\
  --model point_transformer \\
  --dataset stanford3d \\
  --data_dir /path/to/this/dataset
```

## 数据来源

- **原始数据集**: Stanford3dDataset_v1.2
- **原始URL**: http://buildingparser.stanford.edu/dataset.html
- **预处理代码**: https://github.com/ybyyb1/pointcloud-classification

## 引用

如果使用此数据集，请引用原始Stanford3D论文：

```bibtex
@inproceedings{{armeni20163d,
  title={{3D Semantic Parsing of Large-Scale Indoor Spaces}},
  author={{Armeni, Iro and Sener, Ozan and Zamir, Amir R and Jiang, Helen and Brilakis, Ioannis and Fischer, Martin and Savarese, Silvio}},
  booktitle={{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}},
  pages={{1534--1543}},
  year={{2016}}
}}
```

## 许可证

原始Stanford3D数据集有其自己的使用条款。此预处理数据集仅用于研究目的。
"""

    return readme


def create_kaggle_metadata(processed_path: Path) -> dict:
    """创建Kaggle数据集元数据"""
    # 读取原始元数据获取信息
    metadata_path = processed_path / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        original_metadata = json.load(f)

    return {
        "title": "Stanford3D Preprocessed Dataset",
        "id": "ybyyb1/stanford3d-processed",
        "licenses": [{
            "name": "CC0-1.0"
        }],
        "description": "Stanford3D Dataset v1.2 preprocessed for point cloud object classification. Compatible with ScanObjectNN format.",
        "keywords": [
            "point-cloud",
            "3d-vision",
            "computer-vision",
            "stanford3d",
            "object-classification"
        ],
        "isPrivate": False,
        "collaborators": [],
        "data": []
    }


def main():
    parser = argparse.ArgumentParser(description='打包Stanford3D预处理数据用于Kaggle上传')
    parser.add_argument('--input', '-i', default='./data/stanford3d/processed',
                        help='预处理数据目录路径')
    parser.add_argument('--output', '-o', default='./stanford3d_processed_kaggle.zip',
                        help='输出的zip文件路径')
    parser.add_argument('--include-npz', action='store_true',
                        help='包含NPZ文件（会增加文件大小）')
    parser.add_argument('--no-readme', action='store_true',
                        help='不创建README文件')

    args = parser.parse_args()

    try:
        package_stanford3d_for_kaggle(
            processed_dir=args.input,
            output_zip=args.output,
            include_npz=args.include_npz,
            create_readme=not args.no_readme
        )
        return 0
    except Exception as e:
        print(f"打包失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())