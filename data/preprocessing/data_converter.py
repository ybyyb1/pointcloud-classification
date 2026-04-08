"""
数据格式转换
包含各种点云数据格式的转换工具
"""

import numpy as np
import os
import h5py
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import torch


class DataConverter:
    """数据格式转换类"""

    @staticmethod
    def convert_numpy_to_tensor(points: np.ndarray) -> torch.Tensor:
        """
        将numpy数组转换为PyTorch张量

        Args:
            points: numpy数组，形状为 (N, 3) 或 (B, N, 3)

        Returns:
            torch.Tensor: PyTorch张量
        """
        return torch.from_numpy(points).float()

    @staticmethod
    def convert_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        将PyTorch张量转换为numpy数组

        Args:
            tensor: PyTorch张量

        Returns:
            np.ndarray: numpy数组
        """
        return tensor.detach().cpu().numpy()

    @staticmethod
    def save_numpy_to_npz(data: Dict[str, np.ndarray], filepath: str, compressed: bool = True) -> None:
        """
        将numpy数组保存为npz文件

        Args:
            data: 数据字典
            filepath: 文件路径
            compressed: 是否压缩
        """
        if compressed:
            np.savez_compressed(filepath, **data)
        else:
            np.savez(filepath, **data)

    @staticmethod
    def load_npz(filepath: str) -> Dict[str, np.ndarray]:
        """
        加载npz文件

        Args:
            filepath: 文件路径

        Returns:
            Dict[str, np.ndarray]: 数据字典
        """
        return dict(np.load(filepath, allow_pickle=True))

    @staticmethod
    def save_to_h5(data: Dict[str, np.ndarray], filepath: str, compression: str = "gzip") -> None:
        """
        将数据保存为h5文件

        Args:
            data: 数据字典
            filepath: 文件路径
            compression: 压缩方式，可选 "gzip", "lzf", None
        """
        with h5py.File(filepath, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value, compression=compression)

    @staticmethod
    def load_from_h5(filepath: str) -> Dict[str, np.ndarray]:
        """
        从h5文件加载数据

        Args:
            filepath: 文件路径

        Returns:
            Dict[str, np.ndarray]: 数据字典
        """
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
        return data

    @staticmethod
    def save_to_pickle(data: Any, filepath: str) -> None:
        """
        将数据保存为pickle文件

        Args:
            data: 任意Python对象
            filepath: 文件路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_from_pickle(filepath: str) -> Any:
        """
        从pickle文件加载数据

        Args:
            filepath: 文件路径

        Returns:
            Any: Python对象
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_to_json(data: Dict[str, Any], filepath: str) -> None:
        """
        将数据保存为JSON文件

        Args:
            data: 数据字典
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_from_json(filepath: str) -> Dict[str, Any]:
        """
        从JSON文件加载数据

        Args:
            filepath: 文件路径

        Returns:
            Dict[str, Any]: 数据字典
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def convert_ply_to_numpy(ply_filepath: str) -> np.ndarray:
        """
        将PLY文件转换为numpy数组（简化版本，仅支持ASCII格式）

        Args:
            ply_filepath: PLY文件路径

        Returns:
            np.ndarray: 点云数据，形状为 (N, 3)
        """
        points = []

        try:
            with open(ply_filepath, 'r') as f:
                lines = f.readlines()

            # 寻找顶点数据开始位置
            start_index = 0
            for i, line in enumerate(lines):
                if "end_header" in line:
                    start_index = i + 1
                    break

            # 解析顶点数据
            for line in lines[start_index:]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        points.append([x, y, z])
                    except ValueError:
                        continue

        except Exception as e:
            print(f"读取PLY文件失败: {e}")
            # 尝试使用open3d（如果可用）
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(ply_filepath)
                points = np.asarray(pcd.points)
                return points
            except ImportError:
                raise ImportError("请安装open3d库以支持PLY文件读取")

        return np.array(points)

    @staticmethod
    def convert_numpy_to_ply(points: np.ndarray, ply_filepath: str) -> None:
        """
        将numpy数组保存为PLY文件

        Args:
            points: 点云数据，形状为 (N, 3)
            ply_filepath: PLY文件路径
        """
        # 尝试使用open3d（如果可用）
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(ply_filepath, pcd)
            return
        except ImportError:
            pass

        # 手动写入PLY文件（ASCII格式）
        with open(ply_filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")

            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    @staticmethod
    def convert_txt_to_numpy(txt_filepath: str, delimiter: str = None) -> np.ndarray:
        """
        将文本文件转换为numpy数组

        Args:
            txt_filepath: 文本文件路径
            delimiter: 分隔符，如果为None则自动检测

        Returns:
            np.ndarray: 点云数据
        """
        return np.loadtxt(txt_filepath, delimiter=delimiter)

    @staticmethod
    def convert_numpy_to_txt(points: np.ndarray, txt_filepath: str, delimiter: str = " ") -> None:
        """
        将numpy数组保存为文本文件

        Args:
            points: 点云数据
            txt_filepath: 文本文件路径
            delimiter: 分隔符
        """
        np.savetxt(txt_filepath, points, delimiter=delimiter)

    @staticmethod
    def batch_convert_format(input_dir: str, output_dir: str,
                             input_format: str, output_format: str,
                             recursive: bool = True) -> None:
        """
        批量转换文件格式

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            input_format: 输入格式，可选 "ply", "txt", "npy", "npz"
            output_format: 输出格式，可选 "ply", "txt", "npy", "npz"
            recursive: 是否递归处理子目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 支持的格式映射
        format_extensions = {
            "ply": ".ply",
            "txt": ".txt",
            "npy": ".npy",
            "npz": ".npz",
            "h5": ".h5",
            "pkl": ".pkl",
            "json": ".json"
        }

        input_ext = format_extensions.get(input_format.lower())
        output_ext = format_extensions.get(output_format.lower())

        if not input_ext or not output_ext:
            raise ValueError(f"不支持的格式: {input_format} 或 {output_format}")

        # 查找文件
        if recursive:
            filepaths = []
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(input_ext):
                        filepaths.append(os.path.join(root, file))
        else:
            filepaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.endswith(input_ext)]

        print(f"找到 {len(filepaths)} 个 {input_format} 文件")

        # 转换文件
        for i, input_file in enumerate(filepaths):
            try:
                # 读取数据
                if input_format == "ply":
                    points = DataConverter.convert_ply_to_numpy(input_file)
                elif input_format == "txt":
                    points = DataConverter.convert_txt_to_numpy(input_file)
                elif input_format == "npy":
                    points = np.load(input_file)
                elif input_format == "npz":
                    data = DataConverter.load_npz(input_file)
                    # 假设点云数据在"points"键中
                    points = data.get("points")
                    if points is None:
                        print(f"警告: {input_file} 中没有'points'键")
                        continue
                else:
                    raise ValueError(f"不支持的输入格式: {input_format}")

                # 生成输出路径
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + output_ext)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # 保存数据
                if output_format == "ply":
                    DataConverter.convert_numpy_to_ply(points, output_file)
                elif output_format == "txt":
                    DataConverter.convert_numpy_to_txt(points, output_file)
                elif output_format == "npy":
                    np.save(output_file, points)
                elif output_format == "npz":
                    DataConverter.save_numpy_to_npz({"points": points}, output_file)
                else:
                    raise ValueError(f"不支持的输出格式: {output_format}")

                if (i + 1) % 10 == 0:
                    print(f"已转换 {i + 1}/{len(filepaths)} 个文件")

            except Exception as e:
                print(f"转换文件 {input_file} 失败: {e}")

        print(f"批量转换完成: {len(filepaths)} 个文件已转换")

    @staticmethod
    def create_dataset_from_files(filepaths: List[str], labels: List[int],
                                  output_file: str, format: str = "npz") -> None:
        """
        从文件列表创建数据集

        Args:
            filepaths: 文件路径列表
            labels: 标签列表
            output_file: 输出文件路径
            format: 输出格式，可选 "npz", "h5"
        """
        if len(filepaths) != len(labels):
            raise ValueError("文件路径和标签数量不一致")

        all_points = []
        all_labels = []

        for i, (filepath, label) in enumerate(zip(filepaths, labels)):
            try:
                # 根据文件扩展名读取数据
                ext = os.path.splitext(filepath)[1].lower()

                if ext == ".ply":
                    points = DataConverter.convert_ply_to_numpy(filepath)
                elif ext == ".txt":
                    points = DataConverter.convert_txt_to_numpy(filepath)
                elif ext == ".npy":
                    points = np.load(filepath)
                else:
                    print(f"警告: 不支持的文件格式 {ext}，跳过 {filepath}")
                    continue

                all_points.append(points)
                all_labels.append(label)

                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(filepaths)} 个文件")

            except Exception as e:
                print(f"处理文件 {filepath} 失败: {e}")

        # 转换为numpy数组
        all_points_array = np.array(all_points, dtype=object)
        all_labels_array = np.array(all_labels)

        # 保存数据集
        data = {
            "points": all_points_array,
            "labels": all_labels_array
        }

        if format == "npz":
            DataConverter.save_numpy_to_npz(data, output_file)
        elif format == "h5":
            DataConverter.save_to_h5(data, output_file)
        else:
            raise ValueError(f"不支持的输出格式: {format}")

        print(f"数据集创建完成: {len(all_points)} 个样本保存到 {output_file}")


def test_converter():
    """测试数据转换"""
    print("测试数据转换工具...")

    import tempfile
    import shutil

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时目录: {temp_dir}")

    try:
        # 创建测试数据
        points = np.random.randn(100, 3)

        # 测试各种格式转换
        print("\n1. 测试npy格式:")
        npy_file = os.path.join(temp_dir, "test.npy")
        np.save(npy_file, points)
        loaded_points = np.load(npy_file)
        print(f"  保存/加载成功: {loaded_points.shape}")

        print("\n2. 测试npz格式:")
        npz_file = os.path.join(temp_dir, "test.npz")
        DataConverter.save_numpy_to_npz({"points": points}, npz_file)
        data = DataConverter.load_npz(npz_file)
        print(f"  保存/加载成功: {data['points'].shape}")

        print("\n3. 测试h5格式:")
        h5_file = os.path.join(temp_dir, "test.h5")
        DataConverter.save_to_h5({"points": points}, h5_file)
        h5_data = DataConverter.load_from_h5(h5_file)
        print(f"  保存/加载成功: {h5_data['points'].shape}")

        print("\n4. 测试pickle格式:")
        pkl_file = os.path.join(temp_dir, "test.pkl")
        DataConverter.save_to_pickle({"points": points, "info": "test"}, pkl_file)
        pkl_data = DataConverter.load_from_pickle(pkl_file)
        print(f"  保存/加载成功: {pkl_data['points'].shape}")

        print("\n5. 测试JSON格式:")
        json_file = os.path.join(temp_dir, "test.json")
        DataConverter.save_to_json({"points": points.tolist(), "count": 100}, json_file)
        json_data = DataConverter.load_from_json(json_file)
        print(f"  保存/加载成功: 数据点数量={json_data['count']}")

        print("\n6. 测试PyTorch张量转换:")
        tensor = DataConverter.convert_numpy_to_tensor(points)
        print(f"  NumPy -> Tensor: {tensor.shape}, {tensor.dtype}")
        numpy_back = DataConverter.convert_tensor_to_numpy(tensor)
        print(f"  Tensor -> NumPy: {numpy_back.shape}")

        # 测试PLY格式（如果open3d可用）
        try:
            print("\n7. 测试PLY格式:")
            ply_file = os.path.join(temp_dir, "test.ply")
            DataConverter.convert_numpy_to_ply(points, ply_file)
            ply_points = DataConverter.convert_ply_to_numpy(ply_file)
            print(f"  保存/加载成功: {ply_points.shape}")
        except Exception as e:
            print(f"  PLY测试跳过: {e}")

        print("\n8. 测试文本格式:")
        txt_file = os.path.join(temp_dir, "test.txt")
        DataConverter.convert_numpy_to_txt(points, txt_file)
        txt_points = DataConverter.convert_txt_to_numpy(txt_file)
        print(f"  保存/加载成功: {txt_points.shape}")

        print("\n数据转换测试通过!")

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"临时目录已清理")


if __name__ == "__main__":
    test_converter()