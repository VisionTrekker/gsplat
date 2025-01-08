import os
import json
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """递归获取 输入文件夹下所有文件的相对路径"""
    paths = []
    # 遍历path_dir文件夹及其所有子文件夹
    for dp, dn, fn in os.walk(path_dir):    # dp:当前文件夹路径; dn:当前文件夹中所有子文件夹名; fn:当前文件夹中所有文件名
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))    # 添加所有文件的相对路径
    return paths


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir    # 输入文件夹
        self.factor = factor        # 图片下采样的倍率
        self.normalize = normalize  # 是否对齐场景，包括图像位姿和点云
        self.test_every = test_every    # 测试图片的采样频率，每8取1

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)  # 从pycolmap实例化一个SceneManager对象，管理 sparse/0 文件夹中的数据
        manager.load_cameras()  # 读取内参 cameras.bin
        manager.load_images()   # 读取外参 images.bin
        manager.load_points3D() # 读取3D点云 points.bin

        imdata = manager.images # 读取 所有COLMAP图像的 外参矩阵（转换成 W2C）

        w2c_mats = []       # 存储 所有COLMAP图像的 W2C变换矩阵 (4,4)
        camera_ids = []     # 存储 所有COLMAP图像的 相机ID
        Ks_dict = dict()        # 存储 所有COLMAP相机 经图像下采样后的 内参矩阵
        params_dict = dict()    # 存储 所有COLMAP相机的 畸变参数
        imsize_dict = dict()    # 存储 所有COLMAP相机 经图像下采样后的 图像宽、高
        mask_dict = dict()      # 存储 所有COLMAP相机的 mask，默认为 None
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        # 遍历所有COLMAP图像 外参
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0) # 当前COLMAP图像的 W2C的变换矩阵 (4,4)
            w2c_mats.append(w2c)

            # 支持多个相机
            camera_id = im.camera_id    # 当前COLMAP图像对应的 相机ID。每个图像都有一个相机ID，不同图像可对应同一相机
            camera_ids.append(camera_id)

            cam = manager.cameras[camera_id]    # 当前COLMAP图像对应相机的 内参
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            K[:2, :] /= factor      # 原内参矩阵 ==> 图像下采样后的 内参（fx，fy，cx，cy 都要除以 factor）
            Ks_dict[camera_id] = K

            # 获取 当前COLMAP图像对应相机的 畸变参数
            type_ = cam.camera_type     # 相机模型
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)  # 畸变参数
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params

            # 计算 当前COLMAP图像对应相机 经图像下采样后的 图像尺寸
            if isinstance(factor, int):
                imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            elif isinstance(factor, float): # 应对factor为浮点数的情况
                imsize_dict[camera_id] = (int(cam.width / factor), int(cam.height / factor))
            else:
                raise TypeError("[COLMAP Parser] factor must be either an int or a float")

            # 当前COLMAP图像对应相机的 mask，默认为None
            mask_dict[camera_id] = None

        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:    # 无COLMAP信息，报错
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):  # COLMAP相机不是 "SIMPLE_PINHOLE" 或 "PINHOLE"，则警告
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        camtoworlds = np.linalg.inv(w2c_mats)   # 所有COLMAP图像的 C2W变换矩阵 (N,4,4)

        # 所有COLMAP图像名称
        image_names = [imdata[k].name for k in imdata]

        # 根据图像名排序后的 所有COLMAP图像的 图像名、C2W位姿、相机ID（之前的 NeRF 结果是按照文件名排序生成的，确保在相同的测试集上测试指标）
        inds = np.argsort(image_names)  # 对image_names排序后的索引
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # 加载 扩展数据（使用于Bilarf数据集）
        self.extconf = {
            "spiral_radius_scale": 1.0, # 螺旋半径比例因子
            "no_factor_suffix": False,  # 是否去除后缀
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):    # 如果存在扩展数据，则用其更新 extconf
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # 加载 场景边界（只使用于前向场景）
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):    # 如果存在 场景边界 数据，则用其更新 bounds
            self.bounds = np.load(posefile)[:, -2:]

        # 加载 图像
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")             # COLMAP使用的图像文件夹，例：garden/images
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix) # GS要使用的图像文件夹，例：garden/images_2
        # 检查图像文件夹是否存在
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        colmap_files = sorted(_get_rel_paths(colmap_image_dir)) # garden/images   下所有图像的 相对路径
        image_files = sorted(_get_rel_paths(image_dir))         # garden/images_2 下所有图像的 相对路径
        colmap_to_image = dict(zip(colmap_files, image_files))

        # GS图像路径：COLMAP图像名（实际在images中存在的）映射到 GS图像路径，如garden/images_2/000000.jpg
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names if os.path.exists(os.path.join(colmap_image_dir, f))]
        # 检查 所有计算的COLMAP图像 与 实际存在的GS图像 数量是否一致
        assert len(image_paths) == len(image_names), "len(image_paths) != len(image_names), some images are missing in the COLMAP image folder."

        # 所有3D点的 世界坐标、重投影误差、RGB颜色
        points = manager.points3D.astype(np.float32)    # 所有3D点的 世界坐标 (M, 3)
        points_err = manager.point3D_errors.astype(np.float32)  # 所有3D点的 重投影误差 (M,)
        points_rgb = manager.point3D_colors.astype(np.uint8)    # 所有3D点的 RGB颜色 (M, 3)

        # 图像名 -> 看到的所有3D点在点云中的索引，{image_name -> [point_idx]}
        point_indices = dict()  # {"image_name_1.jpg": [point_idx_1, point_idx_2, ...], "image_name_2.jpg": [point_idx_3, point_idx_4, ...], ...}
        # 图像ID -> 图像名
        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}  # manager.name_to_image_id：图像名 -> 图像ID
        # 遍历每个3D点，获取观测到该3D点的 图像数据
        for point_id, data in manager.point3D_id_to_images.items(): # manager.point3D_id_to_images：3D点ID -> List存储着：看到该3D点的图像IDs 及 该3D点在对应图像的2D点索引
            # 遍历每个观测到该3D点的图像，将该3D点在点云中的索引添加到 对应图像的列表中
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]     # 3D点ID -> 点云中3D点的索引，用于直接访问点云数据
                point_indices.setdefault(image_name, []).append(point_idx)  # 在 point_indices 中为每个 图像名 初始化一个空list，并添加该3D点在点云中的索引
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()   # 将3D点索引列表 转换为 NumPy数组，并指定数据类型为 int32，提高存储效率和兼容性
        }

        # 归一化场景
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)   # 场景归一化到单位尺度的 相似变换矩阵（场景的up轴对齐到世界Z轴，场景中心平移到坐标原点，归一化场景大小到单位尺度），(4,4)
            camtoworlds = transform_cameras(T1, camtoworlds)    # 变换所有相机位姿
            points = transform_points(T1, points)               # 变换所有3D点位姿

            T2 = align_principle_axes(points)   # 点云最大主方向 对齐到 世界坐标系Z轴的 仿射变换矩阵
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1     # 最终的变换矩阵
        else:
            transform = np.eye(4)

        self.image_names = image_names  # 所有COLMAP图像的 图像名（根据图像名排序后的 ）                      List[str], (num_images,)
        self.image_paths = image_paths  # 所有GS图像的 相对路径，例 garden/images_2/000000.jpg             List[str], (num_images,)
        self.camtoworlds = camtoworlds  # 所有COLMAP图像的 C2W变换矩阵（根据图像名排序后的及 可能归一化后的）     np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids    # 所有COLMAP图像的 相机ID（根据图像名排序后的，多个图像可对应同一相机ID）  List[int], (num_images,)
        self.Ks_dict = Ks_dict          # 所有COLMAP相机 经图像下采样后 去畸变后的 内参矩阵                    Dict of camera_id -> K
        self.params_dict = params_dict  # 所有COLMAP相机的 畸变参数                                        Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # 所有COLMAP相机 经图像下采样后 去畸变后的 图像宽、高                   Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict      # 所有COLMAP相机 经去畸变后的 mask，默认为None                       Dict of camera_id -> mask
        self.points = points            # 所有3D点的 世界坐标（可能归一化后的）                               np.ndarray, (num_points, 3)
        self.points_err = points_err    # 所有3D点的 重投影误差                                            np.ndarray, (num_points,)
        self.points_rgb = points_rgb    # 所有3D点的 RGB颜色 (M, 3)                                       np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # 所有COLMAP图像名 -> 观测到的所有3D点在点云中的索引               Dict[image_name, np.ndarray[point_idx1, point_idx2, ...]], [M,]
        self.transform = transform      # 已对所有相机和点云作变换的 变换矩阵                                 np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        # 再根据GS图像的实际尺寸调整所有COLMAP相机的 内参矩阵 和 相机宽、高
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]    # 实际加载的GS图像宽、高
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]  # COLMAP相机 经图像下采样后的 图像宽、高
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width  # K * (W_actual / W_colmap)
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K

            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # 遍历每个COLMAP相机的 畸变参数，计算：畸变映射图，去畸变后的 内参矩阵、图像尺寸、mask
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = fx * x1 * r + width // 2
                mapy = fy * y1 * r + height // 2

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            # 有畸变参数的话
            self.mapx_dict[camera_id] = mapx    # 所有COLMAP相机的 畸变映射图
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist  # 所有COLMAP相机 经图像下采样后 去畸变后的 内参矩阵
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])    # 所有COLMAP相机 经图像下采样后 去畸变后的 图像宽、高
            self.mask_dict[camera_id] = mask    # 所有COLMAP相机 经去畸变后的 mask，默认为None

        # 计算场景大小（相机范围半径）
        camera_locations = camtoworlds[:, :3, 3]    # 所有COLMAP图像的 C2W位置
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1) # 计算所有图像到 场景中心的 欧几里得距离（L2范数）
        self.scene_scale = np.max(dists)    # 场景大小 = 所有图像 到 场景中心 距离的 最大值


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser    # COLMAP Parser
        self.split = split      # 数据划分类型，"train"或"test"
        self.patch_size = patch_size    # 随机裁剪的 图像尺寸
        self.load_depths = load_depths  # 是否使用 depth_loss，需将3D点投影到图像平面 获取GT深度

        indices = np.arange(len(self.parser.image_paths))   # 所有GS图像的 相对路径（Parser中已检查 所有计算的COLMAP图像 与 实际存在的GS图像数量一致）
        if split == "train":    # 训练 Dataset
            # self.indices = indices[indices % self.parser.test_every != 0]
            self.indices = indices  # 使用全部图像训练
        else:   # 测试 Dataset
            self.indices = indices[indices % self.parser.test_every == 0]
            # self.indices = [indices for indices in range(5, 30, 5)] # 与原始3DGS测试一致
        print("Total images: {}, [{}] images: {}".format(len(indices), split, len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]     # 读取该张GS图像
        camera_id = self.parser.camera_ids[index]       # 当前图像的 相机ID
        camtoworlds = self.parser.camtoworlds[index]    # 当前图像的 C2W位姿
        K = self.parser.Ks_dict[camera_id].copy()       # 对应相机的 经图像下采样后 去畸变后的 内参矩阵
        params = self.parser.params_dict[camera_id]     # 对应相机的 畸变参数
        mask = self.parser.mask_dict[camera_id]         # 对应相机的 经去畸变后的 mask

        if len(params) > 0:     # 存在畸变参数，对图像去畸变
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:     # 设定了patch_size，则根据其尺寸随机裁剪图像
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            # 主点 cx, cy 也相应改变
            K[0, 2] -= x
            K[1, 2] -= y

        # 封装数据，转为tensor：相机内参矩阵、C2W位姿、图像、该图像在Dataset中的索引、该图像观测到的3D点的 像素坐标、对应像素的深度
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:    # 如果存在mask，则将其转换为bool类型的tensor
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:    # 若使用 depth_loss，则将3D点云投影到图像平面 获取其深度
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]   # 当前图像观测到的 所有3D点在点云中的索引
            points_world = self.parser.points[point_indices]        # 当前图像观测到的 3D点
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T    # 相机坐标系
            points_proj = (K @ points_cam.T).T      # 归一化平面
            points = points_proj[:, :2] / points_proj[:, 2:3]  # 像素平面坐标，(M, 2)
            depths = points_cam[:, 2]       # 深度值，(M,)
            # 筛选在图像范围内 且 深度>0 的点
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()   # 当前图像观测到的3D点的 像素坐标
            data["depths"] = torch.from_numpy(depths).float()   # 对应像素的 深度

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
