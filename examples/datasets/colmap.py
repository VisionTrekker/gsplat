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

        manager = SceneManager(colmap_dir)  # 实例化一个SceneManager对象，管理COLMAP项目 colmap_dir 中的文件和数据
        manager.load_cameras()  # 读取内参 cameras.bin
        manager.load_images()   # 读取外参 images.bin
        manager.load_points3D() # 读取3D点云 points.bin

        # 读取 所有图像的外参矩阵（转换成 W2C）
        imdata = manager.images
        w2c_mats = []       # 所有图像的 W2C变换矩阵
        camera_ids = []     # 所有图像的 相机ID
        Ks_dict = dict()    # 所有相机 经下采样倍率调整后的 内参矩阵
        params_dict = dict()    # 所有相机 的畸变参数
        imsize_dict = dict()    # 所有相机 经下采样倍率调整后的 宽高
        mask_dict = dict()      # 所有相机 的mask，默认为None
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        # 遍历所有图像外参
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0) # W2C的变换矩阵 (4,4)
            w2c_mats.append(w2c)

            # 支持多个相机
            camera_id = im.camera_id    # 当前图像的相机ID，每个图像都有一个相机ID，可以是相同的相机
            camera_ids.append(camera_id)

            # 获取 当前图像对应相机的内参
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor  # fx，fy，cx，cy 都要除以 factor
            Ks_dict[camera_id] = K

            # 获取 相机畸变参数
            type_ = cam.camera_type     # 当前相机的 模型类型
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
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
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)   # (N,4,4)

        camtoworlds = np.linalg.inv(w2c_mats)   # 所有图像的 C2W变换矩阵

        # 获取所有图片名。不再需要根据图片名对位姿进行排序
        image_names = [imdata[k].name for k in imdata]

        # 之前的 NeRF 结果是按照文件名排序生成的，确保在相同的测试集上报告指标
        # 根据图片名排序后的 图像名、位姿C2W、相机ID
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
        if os.path.exists(extconf_file):
            # 如果存在扩展数据，则用其更新 extconf
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # 加载 场景边界（只使用于前向场景）
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
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

        # 获取GS使用的图像路径
        colmap_files = sorted(_get_rel_paths(colmap_image_dir)) # garden/images下所有图像的 相对路径
        image_files = sorted(_get_rel_paths(image_dir))         # garden/images_2下所有图像的 相对路径
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]    # 重映射图像路径，garden/images_2/000000.jpg

        # 获取 3D点云 和 每个图像中与点云关联的3D点索引{image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)  # 点云的 重投影误差
        points_rgb = manager.point3D_colors.astype(np.uint8)    # 点云的 颜色
        point_indices = dict()  # {"image_name_1.jpg": [point_idx_1, point_idx_2, ...], "image_name_2.jpg": [point_idx_3, point_idx_4, ...], ...}

        # 创建一个字典，将图像ID映射到图像名
        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}  # manager.name_to_image_id 表示图像名与图像ID的映射
        # 遍历每个3D点，获取观测到该3D点的 相关图像数据
        for point_id, data in manager.point3D_id_to_images.items(): # manager.point3D_id_to_images 表示每个3D点 被观测到的图像ID 以及 对应特征点的索引
            # 遍历每个观测到该3D点的图像，将该3D点在点云中的索引添加到 对应图像的列表中
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]     # 将3D点ID 映射到 点云中3D点的索引，用于直接访问点云数据
                point_indices.setdefault(image_name, []).append(point_idx)  # 在point_indices中为每个图像名初始化一个空list，并添加该3D点索引
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()   # 将3D点索引列表 转换为 NumPy数组并指定数据类型为 int32，提高存储效率和兼容性
        }

        # Normalize the world space.
        if normalize:
            # 如果要进行归一化处理，先计算 所有相机坐标轴 与 世界坐标轴 对齐的变换矩阵，再计算点云对齐到其主轴（Z轴）的 变换矩阵，并应用于 所有相机位姿与点云
            T1 = similarity_from_cameras(camtoworlds)   # 计算 所有图像的相机坐标轴 与 世界坐标轴 对齐的变换矩阵，(4,4)
            camtoworlds = transform_cameras(T1, camtoworlds)    # 变换所有相机坐标轴
            points = transform_points(T1, points)   # 变换所有3D点云

            T2 = align_principle_axes(points)   # 计算点云对齐到其主轴（Z轴）的 变换矩阵
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1     # 最终的变换矩阵
        else:
            transform = np.eye(4)

        self.image_names = image_names  # 所有图像的 图片名，List[str], (num_images,)
        self.image_paths = image_paths  # GS要使用的所有图像的 相对路径，例garden/images_2/000000.jpg，List[str], (num_images,)
        self.camtoworlds = camtoworlds  # 变换后的 所有图像的 C2W位姿，np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids    # 所有相机ID，多个图像可以是同一个相机ID，即来自于相同的相机，List[int], (num_images,)
        self.Ks_dict = Ks_dict      # 所有相机 经下采样倍率调整后的 内参矩阵，Dict of camera_id -> K
        self.params_dict = params_dict  # 所有相机的 畸变参数，Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # 所有相机 经下采样倍率调整后的 宽高，Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # 所有相机的 mask，默认为None，Dict of camera_id -> mask
        self.points = points    # 变换后的 所有点云，np.ndarray, (num_points, 3)
        self.points_err = points_err  # 所有点云的 重投影误差，np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # 所有点云的 颜色，np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # 所有图像 与其 对应的所有3D点的索引，Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform    # 变换矩阵，np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
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

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]    # 当前图像的3D点
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T  # 归一化平面
            points = points_proj[:, :2] / points_proj[:, 2:3]  # 像素平面坐标，(M, 2)
            depths = points_cam[:, 2]  # 深度值，(M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

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
