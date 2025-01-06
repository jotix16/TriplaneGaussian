from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from pytorch3d.renderer import (
    AlphaCompositor,
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from tgs.utils.base import BaseModule
from tgs.utils.ops import interpolate_image_tokens, points_projection
from tgs.utils.typing import *

from .SPD import SPD
from .SPD_crossattn import SPD_crossattn
from .SPD_pp import SPD_pp
from .utils import MLP_CONV, fps_subsample

SPD_BLOCK = {
    "SPD": SPD,
    "SPD_crossattn": SPD_crossattn,
    "SPD_PP": SPD_pp,
}


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels=1152,
        dim_feat=512,
        num_p0=512,
        radius=1,
        bounding=True,
        up_factors=None,
        SPD_type="SPD",
        token_type="image_token",
        multiview_feature_reduction="mean",
    ):
        super(Decoder, self).__init__()
        # self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_p0)

        # Literal['min', 'max', 'sum', 'mean', 'prod', "fuse_mean_std_cat"]
        self.multiview_feature_reduction = multiview_feature_reduction

        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors
        uppers = []
        self.num_p0 = num_p0
        self.mlp_feat_cond = MLP_CONV(in_channel=input_channels, layer_dims=[dim_feat * 2, dim_feat])

        dim_feat = dim_feat * (2 if multiview_feature_reduction in ["fuse_mean_std_cat", "concat"] else 1)
        for i, factor in enumerate(up_factors):
            uppers.append(
                SPD_BLOCK[SPD_type](
                    dim_feat=dim_feat, up_factor=factor, i=i, bounding=bounding, radius=radius
                )
            )
        self.uppers = nn.ModuleList(uppers)
        self.token_type = token_type

    def calculate_pcl_token(self, pcl_token, up_factor):
        up_token = F.interpolate(pcl_token, scale_factor=up_factor, mode="nearest")
        return up_token

    def calculate_image_token(
        self,
        pcd: Float[Tensor, "B Np 3"],
        input_image_tokens: Float[Tensor, "B Nv C Nt"],
        batch,
    ):
        """
        Args:
            rgb_cond: Tensor, (B, Nv, 3, H, W)
            c2w_cond Tensor, (B, Nv, 4, 4)
            intrinsic_cond, Tensor, (B, Nv, 3, 3)
        """
        h_cond, w_cond = batch["rgb_cond"].shape[-2:]
        c2w_cond = batch["c2w_cond"]
        intrinsic_cond = batch["intrinsic_cond"]
        # batch["c2w_cond"][..., :3, 1:3] *= -1

        local_image_features = interpolate_image_tokens(
            input_image_tokens[..., 1:],  # remove global token
            h_cond,
            w_cond,
            token_patch_size=14,  # dinov2: patch=14x14
        )
        local_features_proj = points_projection(
            pcd,
            c2w_cond,
            intrinsic_cond,
            local_image_features,
            fuse_function=self.multiview_feature_reduction,
        )
        return rearrange(local_features_proj, "B Np C -> B C Np")

    def forward(self, x):
        """
        Args:
            points: Tensor, (B, Np, 3)
            rgb_cond: Tensor, (B, Nv, 3, H, W)
            input_image_tokens: Tensor, (B, Nv, C, Nt) dinov2: 325x768
            c2w_cond Tensor, (B, Nv, 4, 4)
            intrinsic_cond, Tensor, (B, Nv, 3, 3)
        """

        points: Float[Tensor, "B Np 3"] = x["points"]
        if self.token_type == "pcl_token":
            feat_cond = x["pcl_token"]
        elif self.token_type == "image_token":
            feat_cond: Float[Tensor, "B Nv C Nt"] = x["input_image_tokens"]

        feat_cond, ps = pack([feat_cond], "* C Nt")
        feat_cond = self.mlp_feat_cond(feat_cond)
        feat_cond = unpack(feat_cond, ps, "* C Nt")[0]  # (B, Nv, C, Nt)

        arr_pcd = []
        feat_prev = None

        # pcd = torch.permute(points, (0, 2, 1)).contiguous()
        pcd = rearrange(points, "b n c -> b c n").contiguous()
        pcl_up_scale = 1
        for upper in self.uppers:
            if self.token_type == "pcl_token":
                up_cond = self.calculate_pcl_token(feat_cond, pcl_up_scale)
                pcl_up_scale *= upper.up_factor
            elif self.token_type == "image_token":
                up_cond = self.calculate_image_token(points, feat_cond, x)

            pcd, feat_prev = upper(pcd, up_cond, feat_prev)
            points = rearrange(pcd, "b n c -> b c n").contiguous()
            arr_pcd.append(points)
        return arr_pcd


class SnowflakeModelSPDPP(BaseModule):
    """
    apply PC^2 / PCL token to decoder
    """

    @dataclass
    class Config(BaseModule.Config):
        input_channels: int = 1152
        dim_feat: int = 128
        num_p0: int = 512
        radius: float = 1
        bounding: bool = True
        use_fps: bool = True
        up_factors: List[int] = field(default_factory=lambda: [2, 2])
        image_full_token_cond: bool = False
        SPD_type: str = "SPD_PP"
        token_type: str = "pcl_token"
        multiview_feature_reduction: str = "mean"

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.decoder = Decoder(
            input_channels=self.cfg.input_channels,
            dim_feat=self.cfg.dim_feat,
            num_p0=self.cfg.num_p0,
            radius=self.cfg.radius,
            up_factors=self.cfg.up_factors,
            bounding=self.cfg.bounding,
            SPD_type=self.cfg.SPD_type,
            token_type=self.cfg.token_type,
            multiview_feature_reduction=self.cfg.multiview_feature_reduction,
        )

    def forward(self, x):
        results = self.decoder(x)
        return results
