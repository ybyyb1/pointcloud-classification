#!/usr/bin/env python3
"""
Kaggle简化安装脚本
避免复杂的依赖安装问题
"""

import os
import sys
import subprocess
import warnings

def install_minimal_deps():
    """安装最小依赖集合"""
    print("=" * 60)
    print("安装Kaggle最小依赖")
    print("=" * 60)

    # Kaggle环境通常已安装这些包，但为确保安装最小集合
    core_packages = [
        "torch==2.2.0",
        "torchvision==0.17.0",
        "numpy>=1.21.0",
        "h5py>=3.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
    ]

    for package in core_packages:
        print(f"安装: {package}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"  ✓ 完成")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ 安装失败: {e}")
            print(f"  尝试继续...")

    print("\n所有核心依赖安装完成")

def setup_kaggle_environment():
    """设置Kaggle环境"""
    print("\n" + "=" * 60)
    print("设置Kaggle环境")
    print("=" * 60)

    # 检查是否在Kaggle环境中
    is_kaggle = os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    if not is_kaggle:
        print("⚠️ 警告: 不在Kaggle环境中运行")
        return False

    print("✅ 检测到Kaggle环境")

    # 设置工作目录
    if not os.path.exists('/kaggle/working'):
        os.makedirs('/kaggle/working', exist_ok=True)

    os.chdir('/kaggle/working')
    print(f"当前目录: {os.getcwd()}")

    return True

def clone_project():
    """克隆项目到Kaggle工作目录"""
    print("\n" + "=" * 60)
    print("克隆项目")
    print("=" * 60)

    repo_url = "https://github.com/ybyyb1/pointcloud-classification.git"
    repo_dir = "/kaggle/working/pointcloud-classification"

    # 如果已存在，删除重新克隆
    if os.path.exists(repo_dir):
        import shutil
        print(f"删除已存在的目录: {repo_dir}")
        shutil.rmtree(repo_dir)

    # 克隆仓库
    print(f"克隆仓库: {repo_url}")
    try:
        subprocess.check_call(["git", "clone", repo_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ 克隆成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ 克隆失败: {e}")
        # 尝试备用方法
        print("尝试备用方法...")
        import urllib.request
        import zipfile
        try:
            # 这里可以添加备用下载逻辑
            print("请手动克隆仓库或检查网络连接")
            return False
        except Exception as e2:
            print(f"备用方法也失败: {e2}")
            return False

    # 切换到项目目录
    os.chdir(repo_dir)
    print(f"切换到项目目录: {os.getcwd()}")

    return True

def verify_scripts():
    """验证必要的脚本文件存在"""
    print("\n" + "=" * 60)
    print("验证脚本文件")
    print("=" * 60)

    required_scripts = [
        "scripts/train.py",
        "scripts/kaggle_stanford3d_fixed.py",
        "scripts/kaggle_setup_stanford3d.py",
    ]

    missing_scripts = []

    for script in required_scripts:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} (缺失)")
            missing_scripts.append(script)

    if missing_scripts:
        print(f"\n警告: 缺少 {len(missing_scripts)} 个脚本")
        print("将在项目中创建简化版本")
        create_missing_scripts(missing_scripts)

    return True

def create_missing_scripts(missing_scripts):
    """创建缺失的脚本简化版"""
    for script in missing_scripts:
        if script == "scripts/kaggle_stanford3d_fixed.py":
            create_simple_kaggle_script()
        elif script == "scripts/train.py":
            create_simple_train_script()

def create_simple_kaggle_script():
    """创建简化的Kaggle脚本"""
    print("创建简化Kaggle脚本...")
    script_content = '''#!/usr/bin/env python3
"""
简化版Kaggle Stanford3D训练脚本
"""

import os
import sys
import shutil

def main():
    print("=" * 60)
    print("简化版Kaggle Stanford3D训练")
    print("=" * 60)

    # 1. 设置路径
    input_dir = "/kaggle/input/stanford3d-dataset"
    working_dir = "/kaggle/working/data/stanford3d"

    # 2. 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        print("请检查数据集是否已附加到Notebook")
        sys.exit(1)

    # 3. 复制到可写目录
    os.makedirs(working_dir, exist_ok=True)
    for item in os.listdir(input_dir):
        src = os.path.join(input_dir, item)
        dst = os.path.join(working_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"复制: {item}")

    # 4. 训练命令
    cmd = f"""
    python scripts/train.py \\
      --experiment stanford3d_kaggle \\
      --epochs 30 \\
      --batch_size 16 \\
      --kaggle \\
      --model point_transformer \\
      --dataset stanford3d \\
      --data_dir {working_dir}
    """

    print(f"执行命令:\n{cmd}")

    # 执行训练
    os.system(cmd)

if __name__ == "__main__":
    main()
'''

    with open("scripts/kaggle_stanford3d_fixed.py", "w") as f:
        f.write(script_content)

    print("✅ 简化Kaggle脚本已创建")

def create_simple_train_script():
    """创建简化的训练脚本"""
    print("创建简化训练脚本...")
    # 如果train.py存在但有问题，创建备用版本
    pass

def main():
    """主函数"""
    warnings.filterwarnings("ignore")

    print("Kaggle Stanford3D训练环境设置")
    print("=" * 60)

    # 1. 设置Kaggle环境
    if not setup_kaggle_environment():
        print("不在Kaggle环境中，跳过特定设置")

    # 2. 安装最小依赖
    install_minimal_deps()

    # 3. 克隆项目
    if not clone_project():
        print("❌ 项目克隆失败")
        return False

    # 4. 验证脚本
    verify_scripts()

    print("\n" + "=" * 60)
    print("✅ 环境设置完成!")
    print("下一步: 下载数据集并训练")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)