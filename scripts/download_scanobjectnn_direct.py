#!/usr/bin/env python3
"""
直接下载ScanObjectNN数据集脚本
尝试多种来源获取main_split.h5文件
"""
import os
import sys
import urllib.request
import urllib.error
import shutil
import time
import ssl
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple

# 禁用SSL验证（某些镜像可能需要）
ssl._create_default_https_context = ssl._create_unverified_context

def calculate_md5(filepath: str) -> str:
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_h5_file(filepath: str, min_size_kb: int = 100) -> bool:
    """
    验证h5文件是否有效

    Args:
        filepath: 文件路径
        min_size_kb: 最小文件大小（KB），避免空文件或错误页面

    Returns:
        bool: 文件是否有效
    """
    import os

    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return False

    # 检查文件大小
    file_size_kb = os.path.getsize(filepath) / 1024
    if file_size_kb < min_size_kb:
        print(f"文件太小 ({file_size_kb:.1f}KB)，可能无效: {filepath}")
        return False

    # 检查文件扩展名
    if not filepath.lower().endswith('.h5'):
        print(f"文件不是.h5格式: {filepath}")
        return False

    return True

def try_download_url(url: str, output_path: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    尝试从指定URL下载文件

    Returns:
        Tuple[bool, str]: (是否成功, 错误信息或成功信息)
    """
    try:
        print(f"尝试下载: {url}")

        # 创建下载器
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        ]
        urllib.request.install_opener(opener)

        # 下载文件
        start_time = time.time()
        urllib.request.urlretrieve(url, output_path)
        download_time = time.time() - start_time

        # 验证下载的文件
        if validate_h5_file(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[OK] 下载成功: {file_size_mb:.2f}MB, 耗时: {download_time:.1f}s")
            return True, f"下载成功，文件大小: {file_size_mb:.2f}MB"
        else:
            # 删除无效文件
            if os.path.exists(output_path):
                os.remove(output_path)
            return False, "下载的文件无效"

    except urllib.error.HTTPError as e:
        error_msg = f"HTTP错误 {e.code}: {e.reason}"
        return False, error_msg
    except urllib.error.URLError as e:
        error_msg = f"URL错误: {e.reason}"
        return False, error_msg
    except Exception as e:
        error_msg = f"下载错误: {str(e)}"
        return False, error_msg

def get_download_urls() -> List[str]:
    """
    获取ScanObjectNN数据集的可能下载URL

    Returns:
        List[str]: URL列表
    """
    urls = []

    # 1. 官方GitHub仓库（可能失效）
    base_github_urls = [
        "https://github.com/hkust-vgd/scanobjectnn/raw/master/h5_files/main_split/",
        "https://github.com/hkust-vgd/scanobjectnn/raw/master/h5_files/",
        "https://raw.githubusercontent.com/hkust-vgd/scanobjectnn/master/h5_files/main_split/",
        "https://raw.githubusercontent.com/hkust-vgd/scanobjectnn/master/h5_files/",
    ]

    # 可能的文件名
    filenames = [
        "training_objectdataset.h5",
        "testing_objectdataset.h5",
        "train.h5",
        "test.h5",
        "main_split.h5",
        "scanobjectnn.h5"
    ]

    # 生成所有可能的组合
    for base_url in base_github_urls:
        for filename in filenames:
            urls.append(f"{base_url}{filename}")

    # 2. GitHub镜像
    github_mirrors = [
        "https://hub.fastgit.xyz/hkust-vgd/scanobjectnn/raw/master/h5_files/main_split/",
        "https://raw.fastgit.org/hkust-vgd/scanobjectnn/master/h5_files/main_split/",
        "https://gitcode.net/mirrors/hkust-vgd/scanobjectnn/raw/master/h5_files/main_split/",
    ]

    for mirror in github_mirrors:
        for filename in filenames:
            urls.append(f"{mirror}{filename}")

    # 3. 其他可能的来源
    other_sources = [
        # 添加其他已知的下载源
    ]

    urls.extend(other_sources)

    # 去重
    unique_urls = list(dict.fromkeys(urls))
    print(f"生成 {len(unique_urls)} 个可能的下载URL")

    return unique_urls

def download_with_progress(url: str, output_path: str) -> bool:
    """
    带进度显示的下载函数

    Returns:
        bool: 是否下载成功
    """
    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded / total_size * 100
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"下载进度: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end='\r')

        print(f"开始下载: {url}")
        urllib.request.urlretrieve(url, output_path, report_progress)
        print("\n下载完成!")
        return True

    except Exception as e:
        print(f"\n下载失败: {e}")
        return False

def download_from_multiple_sources(output_dir: str = "./data/scanobjectnn",
                                 filename: str = "main_split.h5") -> bool:
    """
    从多个源尝试下载ScanObjectNN数据集

    Args:
        output_dir: 输出目录
        filename: 输出文件名

    Returns:
        bool: 是否下载成功
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # 检查是否已存在有效文件
    if os.path.exists(output_path) and validate_h5_file(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] 文件已存在且有效: {output_path} ({file_size_mb:.2f}MB)")
        return True

    # 获取所有可能的URL
    urls = get_download_urls()

    print("=" * 80)
    print(f"ScanObjectNN数据集下载工具")
    print(f"目标文件: {output_path}")
    print(f"将尝试 {len(urls)} 个可能的下载源")
    print("=" * 80)

    # 尝试每个URL
    success = False
    last_error = ""

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] 尝试源 {i}")

        # 临时文件名
        temp_path = output_path + ".tmp"

        # 尝试下载
        downloaded, message = try_download_url(url, temp_path)

        if downloaded:
            # 重命名为最终文件
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(temp_path, output_path)

            # 计算MD5（可选）
            try:
                md5_hash = calculate_md5(output_path)
                print(f"文件MD5: {md5_hash}")
            except:
                pass

            success = True
            break
        else:
            print(f"X 失败: {message}")
            last_error = message

            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 结果报告
    print("\n" + "=" * 80)
    if success:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] 下载成功! 文件保存在: {output_path}")
        print(f"   文件大小: {file_size_mb:.2f} MB")
        print(f"   下载源: {url}")

        # 测试文件是否可以打开
        try:
            import h5py
            with h5py.File(output_path, 'r') as f:
                print(f"   文件结构: {list(f.keys())}")
                if 'train_points' in f:
                    train_shape = f['train_points'].shape
                    test_shape = f['test_points'].shape if 'test_points' in f else "N/A"
                    print(f"   训练数据: {train_shape}")
                    print(f"   测试数据: {test_shape}")
        except Exception as e:
            print(f"   警告: 无法验证H5文件内容: {e}")

        return True
    else:
        print(f"[X] 所有下载源都失败了")
        print(f"最后错误: {last_error}")
        print("\n备选方案:")
        print("1. 手动下载:")
        print("   - 访问: https://github.com/hkust-vgd/scanobjectnn")
        print("   - 下载main_split.h5文件")
        print("   - 保存到: " + output_path)
        print("2. 使用Kaggle数据集:")
        print("   - 在Kaggle上搜索: hkustvgd/scanobjectnn")
        print("   - 下载并解压到: " + output_dir)
        print("3. 使用虚拟数据集测试:")
        print("   - 设置环境变量: SCANOBJECTNN_ALLOW_DUMMY=true")
        print("   - 系统会自动生成测试数据")

        return False

def create_sample_download_script():
    """创建样本下载脚本（用于手动下载）"""
    script_content = """#!/bin/bash
# ScanObjectNN手动下载脚本
# 如果自动下载失败，可以使用此脚本手动下载

set -e

DATA_DIR="./data/scanobjectnn"
mkdir -p "$DATA_DIR"

echo "请选择下载方式:"
echo "1. 从Kaggle下载（需要kaggle API）"
echo "2. 从Google Drive下载（需要gdown）"
echo "3. 其他来源"
read -p "请输入选项 (1-3): " choice

case $choice in
    1)
        echo "从Kaggle下载..."
        kaggle datasets download -d hkustvgd/scanobjectnn
        unzip scanobjectnn.zip -d "$DATA_DIR"
        rm scanobjectnn.zip
        ;;
    2)
        echo "从Google Drive下载..."
        pip install gdown
        # 这里需要实际的Google Drive文件ID
        gdown --id "YOUR_GOOGLE_DRIVE_FILE_ID" -O "$DATA_DIR/main_split.h5"
        ;;
    3)
        echo "请手动下载main_split.h5并保存到: $DATA_DIR/"
        echo "下载地址可能包括:"
        echo "- https://github.com/hkust-vgd/scanobjectnn"
        echo "- 其他镜像站点"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo "下载完成!"
"""

    script_path = "./scripts/manual_download_scanobjectnn.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"手动下载脚本已创建: {script_path}")
    return script_path

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ScanObjectNN数据集下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认下载到 ./data/scanobjectnn/main_split.h5
  python download_scanobjectnn_direct.py

  # 指定输出目录
  python download_scanobjectnn_direct.py --output_dir ./my_data

  # 指定文件名
  python download_scanobjectnn_direct.py --filename scanobjectnn_data.h5

  # 创建手动下载脚本
  python download_scanobjectnn_direct.py --create_script
        """
    )

    parser.add_argument("--output_dir", type=str, default="./data/scanobjectnn",
                       help="输出目录")
    parser.add_argument("--filename", type=str, default="main_split.h5",
                       help="输出文件名")
    parser.add_argument("--create_script", action="store_true",
                       help="创建手动下载脚本")

    args = parser.parse_args()

    if args.create_script:
        script_path = create_sample_download_script()
        print(f"手动下载脚本已创建: {script_path}")
        return

    print("ScanObjectNN数据集直接下载工具")
    print(f"输出目录: {args.output_dir}")
    print(f"输出文件: {args.filename}")
    print()

    # 执行下载
    success = download_from_multiple_sources(args.output_dir, args.filename)

    if success:
        print("\n[OK] 数据集下载完成!")
        print(f"文件位置: {os.path.join(args.output_dir, args.filename)}")
        print("\n下一步:")
        print("1. 验证数据集: python test_dataset_fix.py")
        print("2. 训练模型: python main.py train --model point_transformer --dataset scanobjectnn")
    else:
        print("\n[X] 数据集下载失败")
        sys.exit(1)

if __name__ == "__main__":
    main()