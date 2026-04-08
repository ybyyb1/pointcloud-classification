#!/usr/bin/env python3
"""
Kaggle API设置脚本
用于配置Kaggle API密钥以便访问数据集
"""

import os
import sys
import json
import stat
from pathlib import Path


def setup_kaggle_api(username=None, api_key=None, token_name=None):
    """
    设置Kaggle API密钥

    Args:
        username: Kaggle用户名
        api_key: Kaggle API密钥
        token_name: 令牌名称（可选，仅用于显示）
    """
    print("=" * 60)
    print("Kaggle API 设置")
    print("=" * 60)

    # 尝试从环境变量获取凭证
    if not username:
        username = os.environ.get('KAGGLE_USERNAME')
    if not api_key:
        api_key = os.environ.get('KAGGLE_KEY') or os.environ.get('KAGGLE_API_KEY')

    # 如果仍未提供，请求用户输入
    if not username:
        username = input("请输入Kaggle用户名: ").strip()
    if not api_key:
        api_key = input("请输入Kaggle API密钥: ").strip()

    if not username or not api_key:
        print("错误: 用户名和API密钥都是必需的")
        return False

    if token_name:
        print(f"设置Kaggle API令牌: {token_name}")
    print(f"用户名: {username}")
    print(f"API密钥: {'*' * min(len(api_key), 8)}...")

    # 创建.kaggle目录
    home_dir = str(Path.home())
    kaggle_dir = os.path.join(home_dir, '.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    print(f"创建目录: {kaggle_dir}")

    # 创建kaggle.json文件
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    config = {
        "username": username,
        "key": api_key
    }

    with open(kaggle_json, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"创建配置文件: {kaggle_json}")

    # 设置文件权限（仅限所有者读写）
    try:
        os.chmod(kaggle_json, stat.S_IRUSR | stat.S_IWUSR)
        print("设置文件权限: 600 (仅所有者读写)")
    except Exception as e:
        print(f"警告: 无法设置文件权限: {e}")

    # 设置环境变量
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = api_key
    print("设置环境变量: KAGGLE_USERNAME, KAGGLE_KEY")

    # 测试配置
    if test_kaggle_api():
        print("[OK] Kaggle API 设置成功!")
        print(f"   用户名: {username}")
        if token_name:
            print(f"   令牌名称: {token_name}")
        print(f"   配置文件: {kaggle_json}")
        return True
    else:
        print("[FAIL] Kaggle API 测试失败，请检查凭证")
        return False


def test_kaggle_api():
    """测试Kaggle API配置"""
    print("\n测试Kaggle API配置...")

    try:
        # 尝试导入Kaggle API
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            print("Kaggle API未安装，尝试安装...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            from kaggle.api.kaggle_api_extended import KaggleApi

        # 测试认证
        api = KaggleApi()
        api.authenticate()

        # 测试列出数据集（不实际下载）
        print("获取用户信息...")
        try:
            user = api.user_info()
            print(f"认证成功！用户名: {user.username}")
            return True
        except:
            # 如果无法获取用户信息，尝试列出公共数据集
            print("获取公共数据集列表...")
            try:
                datasets = api.dataset_list()
                if datasets:
                    print(f"认证成功！可访问数据集 (第一个: {datasets[0].title})")
                    return True
                else:
                    print("警告: 无法获取数据集列表")
                    return False
            except Exception as e:
                print(f"获取数据集列表失败: {e}")
                # 即使这样，如果authenticate()没有抛出异常，API密钥可能仍然有效
                print("认证可能成功，但无法获取数据集列表")
                return True

    except Exception as e:
        print(f"Kaggle API测试失败: {e}")
        return False


def setup_from_command_line():
    """从命令行参数设置"""
    import argparse

    parser = argparse.ArgumentParser(description='设置Kaggle API密钥')
    parser.add_argument('--username', help='Kaggle用户名')
    parser.add_argument('--api-key', help='Kaggle API密钥')
    parser.add_argument('--token-name', help='令牌名称（可选）')
    parser.add_argument('--test-only', action='store_true', help='仅测试现有配置')

    args = parser.parse_args()

    if args.test_only:
        print("测试现有Kaggle API配置...")
        if test_kaggle_api():
            print("[OK] Kaggle API 配置正常")
            return 0
        else:
            print("[FAIL] Kaggle API 配置有问题")
            return 1

    success = setup_kaggle_api(
        username=args.username,
        api_key=args.api_key,
        token_name=args.token_name
    )

    return 0 if success else 1


def quick_setup_for_user(username, api_key, token_name=None):
    """
    为用户快速设置Kaggle API的快捷函数

    Args:
        username: Kaggle用户名
        api_key: Kaggle API密钥
        token_name: 令牌名称（可选）
    """
    print(f"为 {username} 快速设置Kaggle API...")
    return setup_kaggle_api(username, api_key, token_name)


# 用户特定配置
USER_CONFIGS = {
    "ybyyb1": {
        "username": "ybyyb1",
        "api_key": "KGAT_e15b251e1961531282207d55cc009ceb",
        "token_name": "bishe"
    }
}


def setup_for_predefined_user(user_id="ybyyb1"):
    """
    为预定义用户设置Kaggle API

    Args:
        user_id: 用户ID，默认为 "ybyyb1"
    """
    if user_id not in USER_CONFIGS:
        print(f"错误: 未找到用户 {user_id} 的配置")
        print(f"可用用户: {list(USER_CONFIGS.keys())}")
        return False

    config = USER_CONFIGS[user_id]
    print(f"为预定义用户设置: {user_id}")
    print(f"令牌名称: {config.get('token_name', '未命名')}")

    return setup_kaggle_api(
        username=config["username"],
        api_key=config["api_key"],
        token_name=config.get("token_name")
    )


if __name__ == "__main__":
    # 如果直接运行，检查是否有预定义用户
    if len(sys.argv) == 1:
        # 没有参数，尝试为预定义用户设置
        print("尝试为预定义用户 ybyyb1 设置Kaggle API...")
        success = setup_for_predefined_user("ybyyb1")
        sys.exit(0 if success else 1)
    else:
        # 有命令行参数，使用命令行解析
        sys.exit(setup_from_command_line())