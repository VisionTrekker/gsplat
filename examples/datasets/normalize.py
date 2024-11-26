import numpy as np


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    从相机位姿（C2W）中获取一个相似性变换矩阵，用于归一化场景
        c2w: 所有相机的 C2W外参变换矩阵 (N,4,4)
        返回：T (4,4) , scale (float)
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
    # Compute centroid
    centroid = np.median(point_cloud, axis=0)

    # Translate point cloud to centroid
    translated_point_cloud = point_cloud - centroid

    # Compute covariance matrix
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues (descending order) so that the z-axis
    # is the principal axis with the smallest eigenvalue.
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Check orientation of eigenvectors. If the determinant of the eigenvectors is
    # negative, then we need to flip the sign of one of the eigenvectors.
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # Create rotation matrix
    rotation_matrix = eigenvectors.T

    # Create SE(3) matrix (4x4 transformation matrix)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid

    return transform


def transform_points(matrix, points):
    """Transform points using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        points: Nx3 array of points

    Returns:
        Nx3 array of transformed points
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    """Transform cameras using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        camtoworlds: Nx4x4 array of camera-to-world matrices

    Returns:
        Nx4x4 array of transformed camera-to-world matrices
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
