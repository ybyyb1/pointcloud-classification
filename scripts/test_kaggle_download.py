#!/usr/bin/env python3
"""测试Kaggle下载ScanObjectNN数据集"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets.scanobjectnn_dataset import download_from_kaggle

def main():
    dataset_name = "hkustvgd/scanobjectnn"
    version = "main_split"
    output_dir = "./data/scanobjectnn"

    print(f"尝试从Kaggle下载数据集: {dataset_name}")
    print(f"版本: {version}")
    print(f"输出目录: {output_dir}")

    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 下载
        file_path = download_from_kaggle(dataset_name, version, output_dir)
        print(f"下载成功! 文件保存在: {file_path}")

        # 验证文件
        from data.datasets.scanobjectnn_dataset import validate_h5_file
        if validate_h5_file(file_path):
            print("文件验证通过!")
        else:
            print("文件验证失败!")

    except Exception as e:
        print(f"下载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())