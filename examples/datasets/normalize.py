import numpy as np


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    计算 所有相机坐标轴 与 世界坐标轴 对齐的变换矩阵
        c2w: 所有相机的 C2W外参变换矩阵 (N,4,4)
    返回：归一化后的 变换矩阵 T，(4,4)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)   # 将所有相机的 相机坐标系up轴（y负轴）转换到世界坐标系中，(N,3,3) -> (N,3)
    world_up = np.mean(ups, axis=0)     # 所有相机up轴 世界坐标的均值，(N,3) -> (3,)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()      # 所有相机up轴 与 世界up轴的 夹角余弦值
    cross = np.cross(world_up, up_camspace) # 所有相机up轴 到 世界up轴的 旋转轴
    # 旋转轴 构造 反对称矩阵
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    # 所有相机up轴 到 世界up轴的 旋转矩阵
    if c > -1:
        # 两轴的夹角 < 180，即旋转角度是可逆的，则可使用 罗德里格斯旋转公式 来计算旋转矩阵
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # 两轴的夹角 = 180，则只需绕x轴旋转180度
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R     # 所有相机的旋转矩阵 应用 世界坐标轴对齐，(3,3) @ (N,3,3) -> (N,3,3)
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1) # 将所有相机的 相机坐标系forward轴（z轴）转换到世界坐标系中，(N,3)
    t = (R_align @ t[..., None])[..., 0]    # 所有相机的位置 应用 世界坐标轴对齐，(3,3) @ (N,3,1) -> (N,3,3) -> (N,3)

    # (2) 重新计算 场景的中心点
    if center_method == "focus":
        # 默认，所有相机视线 与 原点 的最近点 的中位数
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds   # fwds * -t：所有相机中心射线的方向向量，(N,3)
                                                            # .sum(-1)[:, None]：所有相机中心射线到原点的距离，(N,1)
                                                            # t + * fwds：所有相机中心射线到原点的最近点，(N,3)
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        # 所有相机中心位置的中位数
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))  # 1 / 所有相机中心位置到原点距离的最大值或中位数
    transform[:3, :] *= scale

    return transform


def align_principle_axes(point_cloud):
    """计算点云对齐到其主轴（Z轴）的 变换矩阵"""
    centroid = np.median(point_cloud, axis=0)   # 计算点云的质心

    translated_point_cloud = point_cloud - centroid # 将点云平移到以质心为原点

    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)    # 计算平移后的点云的 协方差矩阵，将每一列视为一个变量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)   # 计算协方差矩阵的 特征值和特征向量
    # 按照特征值降序对特征向量排序，以便主轴（特征值最小的轴）成为 z 轴
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # 检查特征向量的方向，如果特征向量的行列式小于 0，则将第一个特征向量的符号取反，以确保特征向量的方向一致。
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # 创建旋转矩阵 = 特征向量的转置
    rotation_matrix = eigenvectors.T

    # 创建 SE(3) 矩阵，（4,4）
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
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
