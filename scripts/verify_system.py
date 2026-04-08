#!/usr/bin/env python3
"""
系统验证脚本
验证点云分类系统是否正常工作
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_python_version():
    """检查Python版本"""
    print_header("Python版本检查")

    import platform
    python_version = platform.python_version()
    python_version_tuple = sys.version_info

    print(f"Python版本: {python_version}")
    print(f"Python详细信息: {sys.version}")

    if python_version_tuple.major == 3 and python_version_tuple.minor >= 8:
        print("✅ Python版本符合要求 (3.8+)")
        return True
    else:
        print("❌ Python版本不符合要求 (需要3.8+)")
        return False


def check_pytorch():
    """检查PyTorch"""
    print_header("PyTorch检查")

    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")

        # 检查CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用: {cuda_available}")

        if cuda_available:
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
            gpu_count = torch.cuda.device_count()
            print(f"GPU数量: {gpu_count}")

            # 检查CUDA版本
            cuda_version = torch.version.cuda
            print(f"CUDA版本: {cuda_version}")
        else:
            print("⚠️  警告: 没有检测到GPU，训练将使用CPU（速度较慢）")

        # 检查基本功能
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y

        print("✅ PyTorch检查通过")
        return True

    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch检查失败: {e}")
        return False


def check_project_structure():
    """检查项目结构"""
    print_header("项目结构检查")

    required_dirs = [
        "config",
        "data",
        "models",
        "training",
        "utils",
        "scripts",
        "docs"
    ]

    required_files = [
        "main.py",
        "requirements.txt",
        "README.md",
        "config/base_config.py",
        "models/base_model.py",
        "training/trainer.py"
    ]

    print("检查目录结构...")
    all_dirs_ok = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ (缺失)")
            all_dirs_ok = False

    print("\n检查关键文件...")
    all_files_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (缺失)")
            all_files_ok = False

    return all_dirs_ok and all_files_ok


def check_imports():
    """检查关键模块导入"""
    print_header("模块导入检查")

    modules_to_check = [
        ("config.base_config", ["SystemConfig", "DatasetConfig", "ModelConfig"]),
        ("models.base_model", ["BaseModel"]),
        ("training.trainer", ["Trainer", "KaggleTrainer"]),
        ("utils.logger", ["setup_logger"]),
        ("data.datasets.base_dataset", ["BaseDataset"]),
    ]

    all_imports_ok = True

    for module_name, classes in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name} 导入成功")

            # 检查类
            for class_name in classes:
                if hasattr(module, class_name):
                    print(f"    ✅ {class_name} 存在")
                else:
                    print(f"    ❌ {class_name} 不存在")
                    all_imports_ok = False

        except ImportError as e:
            print(f"❌ {module_name} 导入失败: {e}")
            all_imports_ok = False
        except Exception as e:
            print(f"❌ {module_name} 检查出错: {e}")
            all_imports_ok = False

    return all_imports_ok


def check_kaggle_integration():
    """检查Kaggle集成"""
    print_header("Kaggle集成检查")

    kaggle_files = [
        "scripts/setup_kaggle.py",
        "scripts/setup_kaggle_github.py",
        "docs/en/kaggle_usage.md",
        "docs/zh-cn/kaggle_usage.md"
    ]

    print("检查Kaggle相关文件...")
    all_files_ok = True

    for file_path in kaggle_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (缺失)")
            all_files_ok = False

    # 检查kaggle.json.example
    kaggle_example = "kaggle.json.example"
    if os.path.exists(kaggle_example):
        print(f"  ✅ {kaggle_example}")
        print(f"    提示: 复制此文件为kaggle.json并填入你的API密钥")
    else:
        print(f"  ⚠️  {kaggle_example} (缺失但可选)")

    return all_files_ok


def check_github_config():
    """检查GitHub配置"""
    print_header("GitHub配置检查")

    # 检查README中的GitHub链接
    readme_path = "README.md"
    github_url = "https://github.com/ybyyb1/pointcloud-classification"

    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if github_url in content:
            print(f"✅ README中包含正确的GitHub链接: {github_url}")
        else:
            print(f"❌ README中未找到GitHub链接: {github_url}")
            print(f"   当前链接应为: {github_url}")
            return False

    except Exception as e:
        print(f"❌ 读取README失败: {e}")
        return False

    return True


def check_training_script():
    """检查训练脚本"""
    print_header("训练脚本检查")

    scripts_to_check = [
        "scripts/train.py",
        "main.py"
    ]

    all_scripts_ok = True

    for script_path in scripts_to_check:
        if os.path.exists(script_path):
            print(f"✅ {script_path} 存在")

            # 检查文件是否可执行
            try:
                with open(script_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("#!/usr/bin/env python"):
                        print(f"    ✅ 可执行脚本")
                    else:
                        print(f"    ⚠️  缺少shebang行")
            except:
                pass

        else:
            print(f"❌ {script_path} 缺失")
            all_scripts_ok = False

    return all_scripts_ok


def run_quick_test():
    """运行快速测试"""
    print_header("快速功能测试")

    tests = [
        ("创建配置对象", "from config.base_config import SystemConfig; config = SystemConfig(); print(f'项目: {config.project_name}')"),
        ("创建虚拟模型", "from models.base_model import BaseModel; import torch; model = BaseModel(); print(f'模型创建成功')"),
        ("检查训练器", "from training.trainer import Trainer; print(f'训练器可用')"),
    ]

    all_tests_passed = True

    for test_name, test_code in tests:
        try:
            exec(test_code, globals())
            print(f"✅ {test_name}")
        except Exception as e:
            print(f"❌ {test_name} 失败: {e}")
            all_tests_passed = False

    return all_tests_passed


def main():
    """主函数"""
    print("=" * 60)
    print("点云分类系统 - 系统验证")
    print("=" * 60)

    print(f"工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.executable}")

    checks = [
        ("Python版本", check_python_version),
        ("项目结构", check_project_structure),
        ("PyTorch", check_pytorch),
        ("模块导入", check_imports),
        ("GitHub配置", check_github_config),
        ("训练脚本", check_training_script),
        ("Kaggle集成", check_kaggle_integration),
        ("快速功能测试", run_quick_test),
    ]

    results = {}

    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"检查 {check_name} 时出错: {e}")
            results[check_name] = False

    # 汇总结果
    print_header("验证结果汇总")

    passed = 0
    total = len(results)

    for check_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name}: {status}")
        if result:
            passed += 1

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n🎉 所有检查通过！系统已准备就绪。")
        print("\n下一步:")
        print("1. 本地使用: python main.py download-scanobjectnn")
        print("2. Kaggle使用: 查看 docs/kaggle_usage.md")
        print("3. 训练模型: python scripts/train.py --experiment test")
        return 0
    else:
        print("\n⚠️  部分检查未通过，请根据上面的提示解决问题。")
        print("\n常见问题解决:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 检查Python版本: 需要Python 3.8+")
        print("3. 安装PyTorch: 访问 https://pytorch.org")
        print("4. 项目结构: 确保在项目根目录运行")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n验证被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n验证过程发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)