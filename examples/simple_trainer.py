import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from gsplat.utils import normalized_quat_to_rotmat
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)
from plyfile import PlyData, PlyElement

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam


@dataclass
class Config:

    disable_viewer: bool = False    # 是否关闭 Viewer

    ckpt: Optional[List[str]] = None    # .pt的读取路径。若提供，则跳过训练，只评测    Optional[X] = None <==> Union[X, None]: 允许变量是 X 或 None

    compression: Optional[Literal["png"]] = None    # 压缩方法名称    Literal[X]: 限制变量的值只能是 X

    render_traj_path: str = "interp"    # 渲染图像的轨迹类型。可选 interp、ellipse、spiral

    data_dir: str = "data/360_v2/garden"    # 输入数据集路径
    data_factor: int = 4    # 图片下采样的倍率
    result_dir: str = "results/garden"      # 结果保存路径

    test_every: int = 8     # 测试图片的采样频率，每8取1

    patch_size: Optional[int] = None    # 随机裁剪的尺寸 (experimental)

    global_scale: float = 1.0           # 场景尺寸的 相对调节因子

    normalize_world_space: bool = True  # Normalize the world space

    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    port: int = 8080    # Viewer的端口号

    batch_size: int = 1     # 训练时每个进程的 batch size，学习率会根据其值自动调整

    steps_scaler: float = 1.0   # 调整迭代次数的 全局因子（倍数）

    max_steps: int = 30_000     # 总迭代次数
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    init_type: str = "sfm"      # 初始化高斯模型的方，sfm 或 random
    init_num_pts: int = 100_000     # 初始高斯个数（随机初始化时有效）
    init_extent: float = 3.0        # 初始高斯的范围倍数（*场景大小，随机初始化时有效）

    sh_degree: int = 3
    sh_degree_interval: int = 1000

    init_opa: float = 0.1       # 初始高斯的 不透明度
    init_scale: float = 1.0     # 初始高斯的 轴长倍数（*3最近邻平均距离）

    ssim_lambda: float = 0.2    # loss中 SSIM的权重

    near_plane: float = 0.01    # 近平面距离
    far_plane: float = 1e10

    # 增稠方法
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )

    packed: bool = False    # 是否在光栅器中使用 “packed 模式”（会降低显存使用量，但是会慢一点。但多卡时因减少GPU之间的数据传输，会加快训练速度）

    sparse_grad: bool = False   # 是否使用 “SparseAdam” 优化器 (experimental)
    visible_adam: bool = False  # 是否使用Taming-3DGS的 “SelectiveAdam” 优化器 (experimental)

    antialiased: bool = False   # 是否在光栅器中使用 “Anti-aliasing”（轻微降低评测指标）

    random_bkgd: bool = False   # 在训练中使用 随机背景颜色（discourage transparency）

    opacity_reg: float = 0.0    # 不透明度 损失权重
    scale_reg: float = 0.0      # 轴长 损失权重

    pose_opt: bool = False      # 是否在训练中优化 训练相机的位姿
    pose_opt_lr: float = 1e-5   # 相机位姿优化的 学习率
    pose_opt_reg: float = 1e-6  # 相机位姿优化正则化的 权重衰减
    pose_noise: float = 0.0     # 是否在相机位姿优化中添加噪声扰动，以该噪声为方差随机初始化Embedding层的权重（仅用于测试 相机位姿优化功能？）

    app_opt: bool = False       # 是否开启光照一致性优化 (experimental)
    app_embed_dim: int = 16     # 光照一致性优化的 “embedding” 维度
    app_opt_lr: float = 1e-3    # 光照一致性优化的 学习率
    app_opt_reg: float = 1e-6   # 光照一致性优化正则化的 权重衰减

    use_bilateral_grid: bool = False    # 是否开启 双边网格优化 “bilateral grid” (experimental)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)    # 双边网格的维度 (X, Y, W)

    depth_loss: bool = False    # 是否使用深度loss (需将3D点投影到图像平面 获取GT深度，experimental)
    depth_lambda: float = 1e-2  # 深度loss的 权重

    tb_every: int = 100     # 每...步将信息写入 tensorboard

    tb_save_image: bool = False     # 是否保存训练图像到 tensorboard

    lpips_net: Literal["vgg", "alex"] = "alex"  # lpips网络名称

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,             # COLMAP Parser
    init_type: str = "sfm",     # 初始化高斯模型时的方法类型，sfm 或 random
    init_num_pts: int = 100_000,    # 初始高斯个数（随机初始化时有效）
    init_extent: float = 3.0,       # 初始高斯的范围倍数（*场景大小，随机初始化时有效）
    init_opacity: float = 0.1,  # 初始高斯的 不透明度
    init_scale: float = 1.0,    # 初始高斯的 轴长倍数
    scene_scale: float = 1.0,   # 场景大小（相机范围半径 * 1.1 * global_scale）
    sh_degree: int = 3,
    sparse_grad: bool = False,  # 是否使用 “SparseAdam” 优化器
    visible_adam: bool = False, # 是否使用Taming-3DGS的 “SelectiveAdam” 优化器
    batch_size: int = 1,    # 每个进程的 batch size，学习率会根据其值自动调整
    feature_dim: Optional[int] = None,  # 若开启光照一致性优化，则额外增加 32个特征维度，否则为None
    device: str = "cuda",   # 指定cuda的编号
    world_rank: int = 0,    # 当前进程的 全局编号（用于分布式训练）
    world_size: int = 1,    # 总进程数
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:

    # 1. 获取初始高斯的 位置、颜色
    if init_type == "sfm":  # 使用 SfM结果 初始化高斯模型
        points = torch.from_numpy(parser.points).float()    # 所有3D点的世界坐标
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()  # 所有3D点的RGB颜色（归一化到 [0,1]）
    elif init_type == "random": # 随机初始化高斯模型，则在 [-init_extent*scene_scale, init_extent*scene_scale] 范围内随机生成 init_num_pts 个点
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # 2. 初始高斯的 轴长（N,3） = 所有点到其3最近邻的 平均距离 * 倍数因子（默认为1.0）
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # 到3个最近邻距离平方的均值，[N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # 3. 将 位置、颜色、轴长 按照进程编号进行 分割，确保每个进程只处理自己的数据（用于Grendel的分布式训练，在单进程模式也起作用）
    points = points[world_rank::world_size]     # 例：world_size=4，world_rank=0处理 0,4,8个高斯，world_rank=1处理 1,5,9个高斯
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    # 4. 随机生成 初始高斯的 旋转四元数，[N, 4]
    quats = torch.rand((N, 4))
    # 5. 初始高斯的 不透明度 = 0.1（逆对数），[N,]
    opacities = torch.logit(torch.full((N,), init_opacity))

    # 6. 添加（位置、轴长、旋转四元数、不透明度）优化参数，并设定 学习率
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),    # lr = 场景大小 * 1.6e-4
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    # 7. 添加（颜色）优化参数
    if feature_dim is None: # 不开启光照一致性优化，则使用SH系数
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:   # 开启，则使用特征向量和点云颜色
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)    # 将优化参数转换成 torch.nn.Parameter 对象，并移动到指定的cuda上

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size    # 总batch大小 = 每个进程的batch大小 * 总进程数
    optimizer_class = None
    if sparse_grad:     # 使用 “SparseAdam” 优化器
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:  # 使用Taming-3DGS的 “SelectiveAdam” 优化器
        optimizer_class = SelectiveAdam
    else:               # 默认使用标准的 Adam 优化器
        optimizer_class = torch.optim.Adam

    # 8. 为每个参数创建一个优化器实例 {"means": torch.optim.Adam(), "scales": torch.optim.Adam(), ...}，并根据 BS 调整学习率、eps、betas
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers   # 返回高斯模型的 优化参数字典 和 对应的优化器字典


class Runner:
    """训练和测试的 Runner"""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        """
        1. 加载数据：创建 COLMAP Parser，训练Dataset，测试Dataset，调整场景大小
        2. 创建初始高斯模型，返回该模型的 优化参数字典 和 对应的优化器字典
        3. 检查参数和优化器是否正确配置
        4. 初始化并返回当前训练策略数据的 state
        5. 创建 高斯模型压缩策略 （可选）
        6. 创建 优化训练相机位姿的 优化器 （可选）
        7. 创建 优化光照一致性的 优化器 （可选）
        8. 创建 优化双边网格的 优化器 （可选）
        9. 创建 SSIM损失 和 PSNR、LPIPS评测对象
        10. 创建 查看器
        """
        set_random_seed(42 + local_rank)

        self.cfg = cfg  # Config对象，包含所有配置参数
        self.world_rank = world_rank    # 当前进程的 全局编号
        self.local_rank = local_rank    # 当前进程在 本节点中的GPU设备编号
        self.world_size = world_size    # 总进程数
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"   # 模型参数 保存文件夹
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"  # 训练（mem、ellipse_time、num_GS）和评测（PNSR、SSIM、LPIPS、ellipse_time、num_GS）结果 保存文件夹
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"   # 在测试迭代次数下 测试集渲染图像 保存文件夹
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # 1. 加载数据
        # 1.1 创建 COLMAP Parser
        #   所有COLMAP图像的 图像名、C2W变换矩阵、相机ID，所有GS图像；    所有COLMAP相机的 内参矩阵、畸变参数、图像宽、高、mask；
        #   所有3D点的 世界坐标、RGB颜色、重投影误差；                  所有COLMAP图像名 -> 观测到的所有3D点在点云中的索引；
        #   场景变换矩阵；     去畸变信息；      场景大小
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        # 1.2 创建 训练Dataset
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        # 1.3 创建 测试Dataset
        self.valset = Dataset(self.parser, split="val")
        # 1.4 调整场景大小（相机范围半径 * 1.1 * global_scale）
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # 2. 创建初始高斯模型，返回该模型的 优化参数字典 和 对应的优化器字典
        feature_dim = 32 if cfg.app_opt else None   # 若开启光照一致性优化，则额外增加 32个特征维度
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # 3. 检查参数和优化器是否正确配置：优化器中的参数 必须和 高斯模型的需计算梯度的参数一一对应，每个参数对应的优化器有且只有一个 param_group，不同的策略必须包含哪些参数
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        # 4. 初始化并返回当前训练策略数据的 state
        if isinstance(self.cfg.strategy, DefaultStrategy):  # 默认策略
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):   # MCMC策略
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # 5. 创建 高斯模型压缩策略
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":    # PNG压缩策略，则使用量化和排序将 splats 压缩成PNG文件；使用 K-means 聚类压缩SH系数
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        # 6. 创建 优化训练相机位姿的 优化器
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)  # 创建 相机位姿优化模型，包含一个Embedding层，每个训练相机的嵌入向量维度为9（3平移 + 6旋转）
            self.pose_adjust.zero_init()    # Embedding层的权重初始化为0

            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:  # 总进程数 > 1，则使用 DDP 分布式并行训练
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:    # 位姿优化噪声 > 0，创建相机位姿优化扰动模型，以该噪声为方差随机初始化Embedding层的权重
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        # 7. 创建 优化光照一致性的 优化器
        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            # 创建 光照一致性优化模型，包含一个Embedding层和一个MLP层
            self.app_module = AppearanceOptModule(len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight) # 将MLP最后一层的权重和偏置初始化为0，以确保初始输出为0
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            # 为2个层分别创建一个优化器
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        # 8. 创建 优化双边网格的 优化器
        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            # 创建 三维双边网格模型，可用于图像的重采样和过滤
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)

            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # 9. 创建 SSIM损失 和 PSNR、LPIPS评测对象
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # 10. 创建 查看器
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        # 光栅器的类型
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        """训练"""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank    # 当前进程的 全局编号
        world_size = self.world_size    # 总进程数

        # 1. 在主进程输出 配置参数
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps   # 总迭代次数
        init_step = 0

        # 2. 创建lr调度器
        # (1) 高斯位置lr的调度器，最终值为 初值的0.01
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:    # (2) 若优化训练相机位姿，则创建相机位姿lr的调度器，最终值为 初值的0.01
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:  # (3) 若优化双边网格，则创建其lr调度器，包含两种方法：前1000代线性增加至lr_init 以warmup；后指数衰减到初值的0.01
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        # 3. 创建 训练集的Dataloader
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)    # 将可迭代对象trainloader转化为 迭代器，可通过next()函数从逐一提取数据，无需遍历整个数据加载器

        # 循环训练
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # 4. 调用Dataset的__getitem__()获取当前迭代的数据（tensor）：
            #   相机内参矩阵 K                        C2W位姿 camtoworld
            #   图像 image                           该图像在Dataset中的索引 image_id
            #   该图像观测到的3D点的 像素坐标 points     对应像素的深度 depths
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)    # [B, 4, 4]
            Ks = data["K"].to(device)       # [B, 3, 3]
            pixels = data["image"].to(device) / 255.0   # [B, H, W, 3]
            num_train_rays_per_step = (pixels.shape[0] * pixels.shape[1] * pixels.shape[2]) # [B,]
            image_ids = data["image_id"].to(device)     # [B,]
            masks = data["mask"].to(device) if "mask" in data else None  # [B, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)      # 当前图像观测到的3D点的 像素坐标，[B, M, 2]
                depths_gt = data["depths"].to(device)   # 对应像素的深度值，[B, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:  # 位姿优化噪声 > 0，则将位姿 输入 位姿扰动模型
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:    # 若要优化相机位姿，则将位姿 输入 位姿优化模型
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # 5. 调整 SH阶数
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # 6. 渲染（前向传播）
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            # 获取 渲染图像 colors， 渲染深度图 depths
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd: # 使用随机背景
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # 7. 保留高斯2D中心的梯度
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # 8. 计算损失
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid")
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.depth_loss:  # 计算深度loss
                # 将可视3D点的 像素坐标 归一化到 [-1,1]
                points = torch.stack([
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ], dim=-1,)   # [B, M, 2]
                grid = points.unsqueeze(2)  # [B, M, 1, 2]
                # 从 渲染深度图 采样，获取对应gt_depth中有效像素的 深度
                depths = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)    # [B, 1, H, W] -> [B, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [B, M]

                # 在 disparity 视差空间中计算loss（视差为深度的倒数）
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [B, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.use_bilateral_grid:  # 计算双边网格loss
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # 正则化
            if cfg.opacity_reg > 0.0:   # 加入 不透明度loss
                loss = (
                    loss
                    + cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0: # 加入 尺度loss
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )
            # 8. 反向传播：计算模型中可训练参数的梯度，存储在参数的.grad属性中
            loss.backward()

            # 9. 打印信息
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            # 在主进程，每 tb_every 步记录一次数据到 tensorboard 中，保存为events.out文件
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:   # 保存 gt图和渲染图 到 tensorboard 中
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # 10. 保存 checkpoint（到达save_steps）
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                # 保存 显存、开始训练到当前的耗时、高斯个数 到.json文件中
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                # 保存 迭代次数、高斯模型 到.ckpt文件中
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # 11. 根据不同的优化器，进行不同特殊操作
            if cfg.sparse_grad:     # (1) 使用 “SparseAdam” 优化器，则将梯度转换为 稀疏张量（在运行优化器之前）
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:    # (2) 使用Taming-3DGS的 “SelectiveAdam” 优化器
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # 12. 更新各模型参数
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()    # 更新模型参数
                optimizer.zero_grad(set_to_none=True)   # 参数（叶子张量）梯度设置为None
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # 13. 调整学习率
            for scheduler in schedulers:
                scheduler.step()

            # 14. 增稠、剪枝（在 backward 和 优化器调整之后）
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # 15. 评测：渲染测试视角图像、生成渲染轨迹视频、保存ply模型（到达eval_steps）
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

                self.save_ply(os.path.join(cfg.result_dir, "point_cloud/iteration_{}.ply".format(step)))

            # 16. 压缩（到达eval_steps 且 要压缩模型）
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    # Experimental
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.splats["sh0"].shape[1]*self.splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.splats["shN"].shape[1]*self.splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # Experimental
    @torch.no_grad()
    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.splats["means"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.splats["opacities"].detach().unsqueeze(-1).cpu().numpy()
        scale = self.splats["scales"].detach().cpu().numpy()
        rotation = self.splats["quats"].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """根据轨迹渲染"""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        # 所有相机的位姿（外参矩阵 C2W，不要开头和结尾的5个）
        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":    # 插值
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 1)  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse": # 椭圆轨迹
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height)  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":  # 螺旋式上升
            camtoworlds_all = generate_spiral_path(camtoworlds_all, bounds=self.parser.bounds * self.scene_scale, spiral_scale_r=self.parser.extconf["spiral_radius_scale"],)
        else:
            raise ValueError(f"Render trajectory type not supported: {cfg.render_traj_path}")

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    """
        local_rank: 当前进程在 本节点中的GPU设备编号
        world_rank: 当前进程的 全局编号（用于分布式训练）
        world_size: 总进程数
        cfg:        Config对象，包含所有配置参数
    """
    if world_size > 1 and not cfg.disable_viewer:   # 分布式运行，则需关闭Viewer（用于渲染和显示），以避免进程间冲突
        cfg.disable_viewer = True
        if world_rank == 0: # 在主进程中打印提示信息
            print("Viewer is disabled in distributed training.")

    # 1. 创建 Runner 实例，用于训练和测试
    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # 2.1 eval
        # 加载ckpt中的权重，并将每个权重的“splats”部分合并到 runner的“splats”中
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        # 评估
        runner.eval(step=step)
        # 加载轨迹渲染图像
        runner.render_traj(step=step)
        # 压缩模型
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        # 2.2 训练
        runner.train()

    # 3. 训练完成后，Viewer等待关闭
    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    # 经命令行输入参数 覆盖后的 configs
    cfg = tyro.extras.overridable_config_cli(configs)   # overridable_config_cli 将默认参数对象 configs 转换成命令行接口（CLI），并将命令行输入的参数 覆盖configs中的默认参数
    # 根据倍数因子 调整 所有迭代次数（当使用4卡时，该值为 0.25。原因：4个GPU相当于batch size为4，因此迭代次数也应相应缩小4倍，以保持整体训练量不变）
    cfg.adjust_steps(cfg.steps_scaler)

    if cfg.compression == "png":    # 若使用 "Png Compression" 压缩方法，则尝试引入额外依赖
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    # 在多节点 多GPU上运行 main 函数
    cli(main, cfg, verbose=True)
