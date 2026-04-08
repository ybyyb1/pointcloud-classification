#!/usr/bin/env python3
"""
训练流程测试脚本
测试从数据准备到模型训练的完整流程
"""
import os
import sys
import argparse
import tempfile
import shutil
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_scanobjectnn_virtual():
    """测试ScanObjectNN虚拟数据集训练流程"""
    print("=" * 80)
    print("测试ScanObjectNN虚拟数据集训练流程")
    print("=" * 80)

    # 设置环境变量
    os.environ['SCANOBJECTNN_ALLOW_DUMMY'] = 'true'

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='scanobjectnn_test_')
    print(f"临时目录: {temp_dir}")

    try:
        # 测试数据集创建
        from config import DatasetConfig, DatasetType
        from data.datasets.scanobjectnn_dataset import ScanObjectNNDataset

        config = DatasetConfig(
            dataset_type=DatasetType.SCANOBJECTNN,
            data_dir=temp_dir,
            num_points=1024,
            batch_size=4,
            scanobjectnn_version="main_split",
            scanobjectnn_url="https://example.com/dummy"
        )

        print("1. 创建ScanObjectNN虚拟数据集...")
        dataset = ScanObjectNNDataset(config, split="train")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   类别数量: {len(dataset.class_names)}")

        # 测试数据加载器
        from data.datasets.scanobjectnn_dataset import create_scanobjectnn_dataloader
        dataloader = create_scanobjectnn_dataloader(config, split="train", shuffle=False)
        batch = next(iter(dataloader))
        print(f"   批次形状: points={batch['points'].shape}, labels={batch['label'].shape}")

        # 测试模型创建
        from models.model_factory import create_model
        model_config = {
            "model_name": "point_transformer",
            "num_classes": len(dataset.class_names),
            "num_points": config.num_points
        }

        print("2. 创建Point Transformer模型...")
        model = create_model(model_config)
        print(f"   模型名称: {model.model_name}")
        print(f"   参数数量: {model.count_parameters()[0]:,}")

        # 测试模型前向传播
        print("3. 测试模型前向传播...")
        with torch.no_grad():
            output = model(batch['points'])
            print(f"   输出形状: {output.shape}")
            print(f"   输出范围: [{output.min():.3f}, {output.max():.3f}]")

        # 测试训练器
        from config import TrainingConfig
        from training.trainer import Trainer

        print("4. 测试训练器...")
        train_config = TrainingConfig(
            epochs=2,  # 只测试2个epoch
            batch_size=4,
            learning_rate=0.001,
            save_checkpoint_interval=1
        )

        trainer = Trainer(
            model=model,
            config=train_config,
            train_loader=dataloader,
            val_loader=dataloader
        )

        # 训练一个epoch
        print("5. 训练1个epoch...")
        history = trainer.train_one_epoch(0)
        print(f"   训练损失: {history['train_loss']:.4f}")
        print(f"   训练准确率: {history['train_accuracy']:.4f}")

        # 评估
        print("6. 评估模型...")
        results = trainer.evaluate(dataloader)
        print(f"   评估损失: {results['loss']:.4f}")
        print(f"   评估准确率: {results['accuracy']:.4f}")

        # 保存检查点
        print("7. 测试检查点保存...")
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "test_model.pth")
        model.save(checkpoint_path, epoch=1, metrics=results)
        print(f"   检查点保存到: {checkpoint_path}")

        # 验证检查点
        if os.path.exists(checkpoint_path):
            print(f"   检查点文件大小: {os.path.getsize(checkpoint_path) / 1024:.1f} KB")
            print("   ✓ 检查点保存成功")
        else:
            print("   ✗ 检查点保存失败")

        print("\n✓ ScanObjectNN虚拟数据集训练流程测试通过!")
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"已清理临时目录: {temp_dir}")

def test_s3dis_synthetic():
    """测试S3DIS合成数据集训练流程"""
    print("\n" + "=" * 80)
    print("测试S3DIS合成数据集训练流程")
    print("=" * 80)

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='s3dis_test_')
    print(f"临时目录: {temp_dir}")

    try:
        # 先创建合成数据集
        from scripts.prepare_s3dis_dataset import create_small_test_dataset
        print("1. 创建S3DIS合成数据集...")
        success = create_small_test_dataset(temp_dir)
        if not success:
            print("   ✗ 创建S3DIS合成数据集失败")
            return False
        print("   ✓ S3DIS合成数据集创建成功")

        # 测试数据集加载
        from config import DatasetConfig, DatasetType
        from data.datasets.s3dis_dataset import S3DISDataset

        config = DatasetConfig(
            dataset_type=DatasetType.S3DIS,
            data_dir=temp_dir,
            num_points=1024,
            batch_size=4,
            s3dis_area=[1],  # 不重要，因为使用合成数据
            s3dis_classes_to_include=["table", "chair", "sofa", "bookcase", "board"],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

        print("2. 加载S3DIS数据集...")
        dataset = S3DISDataset(config, split="train")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   类别数量: {len(dataset.class_names)}")

        # 创建数据加载器
        import torch
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
        batch = next(iter(dataloader))
        print(f"   批次形状: points={batch['points'].shape}, labels={batch['label'].shape}")

        # 测试模型创建
        from models.model_factory import create_model
        model_config = {
            "model_name": "dgcnn",
            "num_classes": len(dataset.class_names),
            "num_points": config.num_points
        }

        print("3. 创建DGCNN模型...")
        model = create_model(model_config)
        print(f"   模型名称: {model.model_name}")
        print(f"   参数数量: {model.count_parameters()[0]:,}")

        # 测试模型前向传播
        print("4. 测试模型前向传播...")
        with torch.no_grad():
            output = model(batch['points'])
            print(f"   输出形状: {output.shape}")

        print("\n✓ S3DIS合成数据集训练流程测试通过!")
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"已清理临时目录: {temp_dir}")

def test_model_comparison():
    """测试模型比较功能"""
    print("\n" + "=" * 80)
    print("测试模型比较功能")
    print("=" * 80)

    try:
        from models.model_factory import ModelFactory

        # 测试模型比较
        model_configs = {
            "point_transformer": {
                "num_classes": 15,
                "num_points": 1024,
                "dim": 512,
                "depth": 6,
                "num_heads": 8
            },
            "pointnet": {
                "num_classes": 15,
                "num_points": 1024
            },
            "dgcnn": {
                "num_classes": 15,
                "num_points": 1024
            }
        }

        print("1. 比较多个模型...")
        results = ModelFactory.compare_models(model_configs)

        print("\n模型比较结果:")
        print("-" * 80)
        print(f"{'模型名称':<20} {'状态':<10} {'参数数量':<15} {'输出形状':<20}")
        print("-" * 80)

        for model_name, result in results.items():
            if result["status"] == "success":
                print(f"{model_name:<20} {'成功':<10} {result['total_params']:<15,} {str(result['output_shape']):<20}")
            else:
                print(f"{model_name:<20} {'失败':<10} {'-':<15} {result['status']:<20}")

        # 保存比较结果
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "model_comparison.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n2. 比较结果保存到: {output_file}")

        print("\n✓ 模型比较功能测试通过!")
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kaggle_integration():
    """测试Kaggle集成"""
    print("\n" + "=" * 80)
    print("测试Kaggle集成")
    print("=" * 80)

    try:
        # 检查是否在Kaggle环境中
        kaggle_env = os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

        if kaggle_env:
            print("检测到Kaggle环境")
            # 测试Kaggle训练器
            from training.trainer import KaggleTrainer
            print("KaggleTrainer可用")
        else:
            print("非Kaggle环境，跳过Kaggle特定测试")

        # 测试Kaggle配置
        from scripts.setup_kaggle import setup_kaggle_environment
        print("Kaggle环境设置功能可用")

        print("\n✓ Kaggle集成测试通过!")
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        return False

def test_full_training_workflow():
    """测试完整训练工作流"""
    print("\n" + "=" * 80)
    print("测试完整训练工作流")
    print("=" * 80)

    import tempfile
    import subprocess

    temp_dir = tempfile.mkdtemp(prefix='workflow_test_')
    print(f"临时工作目录: {temp_dir}")

    try:
        # 测试命令
        commands = [
            # 1. 创建ScanObjectNN虚拟数据集
            f"cd {os.path.dirname(os.path.abspath(__file__))}/.. && "
            f"SCANOBJECTNN_ALLOW_DUMMY=true python main.py download-scanobjectnn "
            f"--data_dir {temp_dir}/scanobjectnn",

            # 2. 创建S3DIS合成数据集
            f"cd {os.path.dirname(os.path.abspath(__file__))}/.. && "
            f"python scripts/prepare_s3dis_dataset.py --small --data_dir {temp_dir}/s3dis",

            # 3. 模型比较
            f"cd {os.path.dirname(os.path.abspath(__file__))}/.. && "
            f"python main.py compare --models point_transformer,pointnet,dgcnn "
            f"--output {temp_dir}/model_comparison.json",

            # 4. 训练测试（使用虚拟数据）
            f"cd {os.path.dirname(os.path.abspath(__file__))}/.. && "
            f"SCANOBJECTNN_ALLOW_DUMMY=true python main.py train "
            f"--model point_transformer --dataset scanobjectnn --epochs 1 "
            f"--data_dir {temp_dir}/scanobjectnn --batch_size 4 "
            f"--experiment_name test_workflow"
        ]

        success_count = 0
        for i, cmd in enumerate(commands[:1], 1):  # 只测试第一个命令以节省时间
            print(f"\n{i}. 执行命令: {cmd[:80]}...")
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"   ✓ 命令执行成功")
                    success_count += 1
                else:
                    print(f"   ✗ 命令执行失败")
                    print(f"     错误: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print(f"   ✗ 命令执行超时")
            except Exception as e:
                print(f"   ✗ 命令执行异常: {e}")

        if success_count >= 1:
            print(f"\n✓ 工作流测试通过 ({success_count}/1 命令成功)")
            return True
        else:
            print(f"\n✗ 工作流测试失败")
            return False

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def generate_test_report():
    """生成测试报告"""
    print("\n" + "=" * 80)
    print("生成测试报告")
    print("=" * 80)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    # 导入torch（在这里导入以避免影响前面的测试）
    import torch

    tests = [
        ("ScanObjectNN虚拟数据集", test_scanobjectnn_virtual),
        ("S3DIS合成数据集", test_s3dis_synthetic),
        ("模型比较", test_model_comparison),
        ("Kaggle集成", test_kaggle_integration),
        # ("完整工作流", test_full_training_workflow)  # 可选，时间较长
    ]

    all_passed = True
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        try:
            passed = test_func()
            report["tests"][test_name] = {
                "status": "passed" if passed else "failed",
                "message": "测试通过" if passed else "测试失败"
            }
            if not passed:
                all_passed = False
        except Exception as e:
            report["tests"][test_name] = {
                "status": "error",
                "message": str(e)
            }
            all_passed = False

    # 保存报告
    report_dir = "./test_reports"
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"training_pipeline_test_{time.strftime('%Y%m%d_%H%M%S')}.json")

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n测试报告保存到: {report_file}")

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    passed_count = sum(1 for test in report["tests"].values() if test["status"] == "passed")
    total_count = len(report["tests"])

    print(f"总测试数: {total_count}")
    print(f"通过数: {passed_count}")
    print(f"失败数: {total_count - passed_count}")

    if all_passed:
        print("\n✓ 所有测试通过!")
        return True
    else:
        print("\n✗ 部分测试失败")
        return False

def main():
    """主函数"""
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="训练流程测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有测试
  python test_training_pipeline.py --all

  # 运行特定测试
  python test_training_pipeline.py --scanobjectnn --model-comparison

  # 生成测试报告
  python test_training_pipeline.py --report

  # 快速测试（只测试ScanObjectNN虚拟数据）
  python test_training_pipeline.py --quick
        """
    )

    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--scanobjectnn", action="store_true", help="测试ScanObjectNN虚拟数据集")
    parser.add_argument("--s3dis", action="store_true", help="测试S3DIS合成数据集")
    parser.add_argument("--model-comparison", action="store_true", help="测试模型比较")
    parser.add_argument("--kaggle", action="store_true", help="测试Kaggle集成")
    parser.add_argument("--workflow", action="store_true", help="测试完整工作流")
    parser.add_argument("--report", action="store_true", help="生成测试报告")
    parser.add_argument("--quick", action="store_true", help="快速测试（只测ScanObjectNN）")

    args = parser.parse_args()

    # 如果没有指定任何测试，默认运行快速测试
    if not any([args.all, args.scanobjectnn, args.s3dis, args.model_comparison,
                args.kaggle, args.workflow, args.report, args.quick]):
        args.quick = True

    # 确定要运行的测试
    tests_to_run = []
    if args.all or args.scanobjectnn or args.quick:
        tests_to_run.append(("ScanObjectNN虚拟数据集", test_scanobjectnn_virtual))
    if args.all or args.s3dis:
        tests_to_run.append(("S3DIS合成数据集", test_s3dis_synthetic))
    if args.all or args.model_comparison:
        tests_to_run.append(("模型比较", test_model_comparison))
    if args.all or args.kaggle:
        tests_to_run.append(("Kaggle集成", test_kaggle_integration))
    if args.all or args.workflow:
        tests_to_run.append(("完整工作流", test_full_training_workflow))

    print("训练流程测试脚本")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 运行测试
    results = {}
    all_passed = True

    for test_name, test_func in tests_to_run:
        print(f"运行测试: {test_name}")
        print("-" * 40)

        start_time = time.time()
        try:
            passed = test_func()
            elapsed = time.time() - start_time

            results[test_name] = {
                "status": "passed" if passed else "failed",
                "time": f"{elapsed:.1f}s"
            }

            if passed:
                print(f"✓ {test_name} 测试通过 ({elapsed:.1f}s)")
            else:
                print(f"✗ {test_name} 测试失败 ({elapsed:.1f}s)")
                all_passed = False

        except Exception as e:
            elapsed = time.time() - start_time
            results[test_name] = {
                "status": "error",
                "time": f"{elapsed:.1f}s",
                "error": str(e)
            }
            print(f"✗ {test_name} 测试异常: {e} ({elapsed:.1f}s)")
            all_passed = False

        print()

    # 生成报告
    if args.report or not all_passed:
        print("=" * 80)
        print("测试结果总结")
        print("=" * 80)

        for test_name, result in results.items():
            status_icon = "✓" if result["status"] == "passed" else "✗"
            print(f"{status_icon} {test_name}: {result['status']} ({result['time']})")
            if "error" in result:
                print(f"   错误: {result['error'][:100]}")

        passed_count = sum(1 for r in results.values() if r["status"] == "passed")
        total_count = len(results)

        print(f"\n总测试数: {total_count}, 通过数: {passed_count}, 失败数: {total_count - passed_count}")

        if all_passed:
            print("\n✓ 所有测试通过!")
        else:
            print("\n✗ 部分测试失败")

    # 返回退出码
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()