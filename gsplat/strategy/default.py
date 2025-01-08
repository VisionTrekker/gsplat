from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split
from typing_extensions import Literal


@dataclass
class DefaultStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005        # 剪枝中的 不透明度阈值，< 该值，则高斯被剪枝。默认为 0.005
    grow_grad2d: float = 0.0002     # 增稠中 高斯中心2D投影位置梯度阈值，> 该值，则被克隆或分裂。默认为 0.0002
    grow_scale3d: float = 0.01      # 增稠中 高斯3D轴长阈值，< 该值*场景尺寸，则被复制；> 该值*场景尺寸，则被分裂。默认为 0.01
    grow_scale2d: float = 0.05      # 增稠中 高斯2D轴长阈值，> 该值*图像分辨率，则被分裂。默认为 0.05
    prune_scale3d: float = 0.1      # 剪枝中的 高斯3D轴长阈值，> 该值*场景尺寸，则被剪枝。默认为 0.1
    prune_scale2d: float = 0.15     # 剪枝中的 高斯2D轴长阈值，> 该值*图像分辨率，则被剪枝。默认为 0.15
    refine_scale2d_stop_iter: int = 0   # 基于高斯2D轴长 增稠的 终止迭代次数。默认为 0
    refine_start_iter: int = 500    # 增稠的 开始迭代次数。默认为 500
    refine_stop_iter: int = 15_000  # 增稠的 终止迭代次数。默认为 15_000
    reset_every: int = 3000         # 重置不透明度的 迭代间隔。默认为 3000
    refine_every: int = 100         # 增稠的 迭代间隔。默认为 100
    pause_refine_after_reset: int = 0   # 重置不透明度后的 暂停增稠的迭代次数。默认为 0，不暂停；有的方法将该值设为 训练集中图像个数
    absgrad: bool = False           # 在分裂高斯时 是否使用 绝对梯度。默认为 False（通常效果会变好，但是需要将 grow_grad2d 升高，例如 0.0008。同时在调用 rasterization时需将 absgrad也设为True）
    revised_opacity: bool = False   # 是否使用 arXiv:2404.06109 论文中提出的 修正复制后的高斯的不透明度（修改为 1 - sqrt(1 - a)）。默认为 False
    verbose: bool = False           # 是否打印详细信息。默认为 False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"   # 使用的增稠方案。3DGS使用 "means2d"梯度、2DGS使用类似的梯度 "gradient_2dgs"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """
        初始化并返回当前训练策略数据的 state ，返回的 state 被传递给 `step_pre_backward()` 和 `step_post_backward()`

        将 state 的初始化推迟到训练的第一步，以确保可以将它们放到正确的设备上
            - grad2d: 每个高斯在图像平面上的 梯度的范数的 累加值
            - count: 每个高斯被训练相机看见的 累加次数
            - radii: 每个高斯的半径（归一化到图像分辨率）
        """
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:   # 如果需要基于高斯2D轴长 增稠，则在 state 中添加 radii
            state["radii"] = None

        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],   # 高斯模型的 优化参数字典
        optimizers: Dict[str, torch.optim.Optimizer],   # 高斯模型的 优化器字典
    ):
        """参数和优化器的 健全性检查

        Check if:
            * `params`和`optimizers`有相同的 keys，即优化器中的参数 必须和 高斯模型的需计算梯度的参数 一一对应
            * 每个参数对应的优化器有且只有一个 param_group
            * 必须存在以下 keys: {"means", "scales", "quats", "opacities"}.
        Raises:
            AssertionError: If any of the above conditions is not met.
        Note:
            这个检查功能不是必须的，但强烈建议在初始化 strategy 后，调用该函数以确保参数和优化器配置正确
        """
        # 1. 优化器中的参数 必须和 高斯模型的需计算梯度的参数 一一对应
        # 2. 每个参数的对应的优化器有且只有一个参数组 param_groups
        super().check_sanity(params, optimizers)
        # 3. 当前增稠策略必须包含 哪些 keys
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],   # 高斯模型参数
        optimizers: Dict[str, torch.optim.Optimizer],   # 其优化器
        state: Dict[str, Any],  # 训练策略数据的 state
        step: int,  # 当前训练的迭代次数
        info: Dict[str, Any],   # 前向传播过程中存储的 数据
    ):
        """回调函数，在`loss.backward()`前执行"""
        # 确保info中包含 高斯的2D中心的梯度
        assert (self.key_for_gradient in info), "The 2D means of the Gaussians is required but missing."
        # 保留该参数（非叶子张量）的梯度
        info[self.key_for_gradient].retain_grad()   # .retain_grad()如果用于保留叶子张量的梯度，在.zero_grad()时仍会被清空

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],   # 高斯模型参数
        optimizers: Dict[str, torch.optim.Optimizer],   # 其优化器
        state: Dict[str, Any],  # 训练策略数据的 state
        step: int,      # 当前训练的迭代次数
        info: Dict[str, Any],   # 前向传播过程中存储的 数据
        packed: bool = False,   # 是否为 packed 模式
    ):
        """回调函数，在在`loss.backward()`后执行"""
        if step >= self.refine_stop_iter:   # > 增稠结束迭代次数，直接返回
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):  # > 增稠开始迭代次数 && 每refine_every次迭代 && > 重置不透明度后需暂停增稠迭代次数
            # 增稠
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # 剪枝
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # 重置训练状态
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0:    # 重置不透明度
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel]  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        # first duplicate
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
