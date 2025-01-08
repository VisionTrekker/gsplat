import numpy as np


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    计算 将场景归一化到单位尺度的 相似变换矩阵（包括旋转、平移、缩放）
        c2w: 所有相机的 C2W位姿， (N,4,4)
        strict_scaling: 是否使用严格缩放（基于最大距离），默认为 False（基于中位数）
        center_method:  用于计算场景中心点的方法，默认为 "focus"
    返回：归一化后的 变换矩阵 T，(4,4)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) 旋转对齐：调整 场景的up轴（所有相机up轴的 世界坐标的 均值）与 世界坐标系Z轴对齐
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)   # 所有相机的up轴（-Y轴）在世界坐标系中的表示，(N,3,3) -> (N,3)
    world_up = np.mean(ups, axis=0)         # 均值，(N,3) -> (3,)
    world_up /= np.linalg.norm(world_up)    # 归一化为单位向量

    up_camspace = np.array([0.0, -1.0, 0.0])    # 相机坐标系中的-Y轴 对应 世界坐标系中的Z轴
    c = (up_camspace * world_up).sum()      # 场景的up轴 与 世界坐标系Z轴 的夹角余弦值
    cross = np.cross(world_up, up_camspace) # 场景的up轴 到 世界坐标系Z轴 的旋转轴
    # 旋转轴 -> 反对称矩阵
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    # 场景的up轴 到 世界坐标系Z轴 的旋转矩阵 R_align
    if c > -1:  # 夹角 < 180°，则使用 罗德里格斯公式 构造旋转矩阵
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:       # 夹角 = 180°（两个方向完全反向），绕 X 轴旋转 180°
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    # 对所有相机的 旋转矩阵、位置 应用坐标轴对齐
    R = R_align @ R     # (3,3) @ (N,3,3) -> (N,3,3)
    t = (R_align @ t[..., None])[..., 0]    # (3,3) @ (N,3,1) -> (N,3,3) -> (N,3)
    # 旋转后的 所有相机的forward轴（Z轴）在世界坐标系中的表示，(N,3)
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)

    # (2) 将场景的几何中心 平移到 坐标原点
    if center_method == "focus":    # 默认，场景几何中心 = 所有相机视线方向到原点的 最近点的 中位数
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds   # fwds * -t：旋转后的 所有相机视线方向 到 原点的 方向向量，(N,3)
                                                            # .sum(-1)[:, None]：所有相机视线方向 到 原点的 距离，(N,1)
                                                            # t + * fwds：所有相机视线方向 到 原点的 最近点，(N,3)
        translate = -np.median(nearest, axis=0)     # 场景几何中心 = 所有最近点的中位数，将场景中心平移到坐标原点
    elif center_method == "poses":  # 所有相机中心位置的 中位数
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate    # 平移部分：将场景中心移动到原点附近
    transform[:3, :3] = R_align     # 旋转部分：对齐 场景的up轴 到 世界坐标系的Z轴

    # (3) 归一化场景大小到 单位尺度
    scale_fn = np.max if strict_scaling else np.median  # 最大值 或 中位数（默认）
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))  # 计算缩放因子 scale = 1 / 旋转平移后的 所有相机中心位置到原点距离的最大值或中位数
    transform[:3, :] *= scale   # 旋转和平移部分同时进行缩放，使场景的范围被归一化到单位尺度

    return transform


def align_principle_axes(point_cloud):
    """
    计算 点云最大主方向 对齐到 世界坐标系Z轴的 仿射变换矩阵
    """
    centroid = np.median(point_cloud, axis=0)   # 点云的几何中心（中位数）

    translated_point_cloud = point_cloud - centroid # 将点云的几何中心 平移到 原点

    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)    # 平移后的点云的 协方差矩阵，描述了点云在各个方向上的分布特性，用于提取主方向
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)       # 协方差矩阵的 特征值（点云沿对应特征向量方向上的分布方差大小）和 特征向量（每个对应点云的一个方向）
    # 将 特征值和特征向量 按特征值大小 降序 排列
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    # 第一列eigenvectors[:,0]为点云的 最大主方向
    # 第二列eigenvectors[:,1]为点云的 次主方向
    # 第三列eigenvectors[:,2]为点云的 最小主方向

    # 检查特征向量是否构成右手系（旋转矩阵是正交阵，其行列式为+1）
    if np.linalg.det(eigenvectors) < 0: # 特征向量的行列式 < 0，构成左手系，则翻转第一个特征向量的方向，确保旋转矩阵是有效的
        eigenvectors[:, 0] *= -1

    # 创建 旋转矩阵 R = 特征向量的转置，R的行向量定义了输出新坐标系的基向量
    # 默认对齐到世界坐标系的Z轴、Y轴、X轴
    rotation_matrix = eigenvectors.T

    # 创建 SE(3) 矩阵，（4,4）
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix     # 旋转部分：对齐 点云最大主方向 到 世界坐标系的Z轴
    transform[:3, 3] = -rotation_matrix @ centroid

    return transform

def transform_points(matrix, points):
    """
    使用 SE(3)变换矩阵 转换 3D点云
        matrix: SE(3)变换矩阵，(4,4)
        points: 3D点云，(N,3)
    返回：变换后的 3D点云，(N,3)
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    """
    使用 SE(3)变换矩阵 转换 相机位姿
        matrix: SE(3)变换矩阵，(4,4)
        camtoworlds: 多个相机位姿（C2W），(N,4,4)
    返回：变换后的 相机位姿，(N,4,4)
    """
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def normalize(camtoworlds, points=None):
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    if points is not None:
        points = transform_points(T1, points)
        T2 = align_principle_axes(points)
        camtoworlds = transform_cameras(T2, camtoworlds)
        points = transform_points(T2, points)
        return camtoworlds, points, T2 @ T1
    else:
        return camtoworlds, T1
