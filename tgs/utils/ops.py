import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from pytorch3d import io
from pytorch3d.renderer import (
    PointsRasterizationSettings, 
    PointsRasterizer)
from pytorch3d.structures import Pointclouds
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
import cv2
from einops import rearrange, repeat, reduce, pack, unpack
from tgs.utils.typing import *

ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]

def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")

def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )

    return directions


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_projection_matrix(
    fovy: Union[float, Float[Tensor, "B"]], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "*B 4 4"]:
    if isinstance(fovy, float):
        proj_mtx = torch.zeros(4, 4, dtype=torch.float32)
        proj_mtx[0, 0] = 1.0 / (math.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[1, 1] = -1.0 / math.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[2, 2] = -(far + near) / (far - near)
        proj_mtx[2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[3, 2] = -1.0
    else:
        batch_size = fovy.shape[0]
        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[:, 1, 1] = -1.0 / torch.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[:, 2, 2] = -(far + near) / (far - near)
        proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(
    c2w: Float[Tensor, "*B 4 4"], proj_mtx: Float[Tensor, "*B 4 4"]
) -> Float[Tensor, "*B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    if c2w.ndim == 2:
        assert proj_mtx.ndim == 2
        w2c: Float[Tensor, "4 4"] = torch.zeros(4, 4).to(c2w)
        w2c[:3, :3] = c2w[:3, :3].permute(1, 0)
        w2c[:3, 3:] = -c2w[:3, :3].permute(1, 0) @ c2w[:3, 3:]
        w2c[3, 3] = 1.0
    else:
        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

def get_intrinsic_from_fov(fov, H, W, bs=-1):
    focal_length = 0.5 * H / np.tan(0.5 * fov)
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = W / 2.0
    intrinsic[1, 2] = H / 2.0

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)

def interpolate_image_tokens(
    image_tokens: Float[Tensor, "B Nv C Nt"],
    h: int,
    w: int,
    token_patch_size: int = 14,
):
    """
    Interpolate image features from shape (B, Nv, C, Nt) to (B, Nv, C, h, w)

    """
    B = image_tokens.shape[0]

    local_features = rearrange(
        image_tokens,
        "B Nv C (H W) -> (B Nv) C H W",
        H=h // token_patch_size,
        W=w // token_patch_size,
    ).contiguous()
    local_features = F.interpolate(
        local_features,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    return rearrange(local_features, "(B Nv) C H W -> B Nv C H W", B=B)

def points_projection(
    points: Float[Tensor, "B NP 3"],
    c2ws: Float[Tensor, "B 4 4"] | Float[Tensor, "B NV 4 4"],
    intrinsics: Float[Tensor, "B 3 3"] | Float[Tensor, "B NV 3 3"],
    local_features: Float[Tensor, "B C H W"] | Float[Tensor, "B NV C H W"],
    # Rasterization settings
    raster_point_radius: float = 0.0075,  # point size
    raster_points_per_pixel: int = 1,  # a single point per pixel, for now
    bin_size: int = 0,
    # ('min', 'max', 'sum', 'mean', 'prod', "fuse_mean_std_cat") or a custom function
    fuse_function: str | Callable[[Tensor, List[int]], Tensor] = "mean",
) -> Float[Tensor, "B NP C"]:

    if c2ws.ndim == 3:  # or (c2ws.ndim == 4 and c2ws.shape[1] == 1):
        # if c2ws.ndim == 4:
        #     c2ws = c2ws.squeeze(1)
        #     intrinsics = intrinsics.squeeze(1)
        #     local_features = local_features.squeeze(1)

        return points_projection_single_view(
            points,
            c2ws,
            intrinsics,
            local_features,
            raster_point_radius,
            raster_points_per_pixel,
            bin_size,
        )

    elif c2ws.ndim == 4:
        return points_projection_multi_view(
            points,
            c2ws,
            intrinsics,
            local_features,
            raster_point_radius,
            raster_points_per_pixel,
            bin_size,
            fuse_function=fuse_function,
        )

    else:
        raise ValueError(f"Invalid c2ws shape, got {c2ws.shape}")

def points_projection_single_view(
    points: Float[Tensor, "B NP 3"],
    c2ws: Float[Tensor, "B 4 4"],
    intrinsics: Float[Tensor, "B 3 3"],
    local_features: Float[Tensor, "B C H W"],
    # Rasterization settings
    raster_point_radius: float = 0.0075,  # point size
    raster_points_per_pixel: int = 1,  # a single point per pixel, for now
    bin_size: int = 0,
):
    B, C, H, W = local_features.shape
    NP = points.shape[1]
    R = raster_points_per_pixel
    device = local_features.device

    # Prepare cameras
    w2cs = torch.inverse(c2ws)
    image_size = repeat(w2cs.new_tensor([H, W]), "hw -> B hw", B=B)
    cameras = cameras_from_opencv_projection(w2cs[:, :3, :3], w2cs[:, :3, 3], intrinsics, image_size)

    # Prepare rasterizer-settings
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=raster_point_radius,
        points_per_pixel=R,
        bin_size=bin_size,
    )

    # Rasterize points
    rasterize = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterize(Pointclouds(points))

    # Visible pixels
    # fragments_idx: point_index for each pixel (-1==no point)
    # visible_pixels: whether the pixel is visible
    # points_to_visible_pixels: point_index for each visible pixel
    fragments_idx: Tensor = fragments.idx.long()
    visible_pixels = fragments_idx > -1  # (B, H, W, R)
    points_to_visible_pixels = fragments_idx[visible_pixels]

    # Reshape local features to (B, H, W, R, C)
    local_features = repeat(local_features, "B C H W -> B H W R C", R=R)

    # Get local features corresponding to visible points
    local_features_proj = torch.zeros(B * NP, C, device=device)
    local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
    local_features_proj = rearrange(local_features_proj, "(B NP) C -> B NP C", B=B).contiguous()

    return local_features_proj

def points_projection_multi_view(
    points: Float[Tensor, "B NP 3"],
    c2ws: Float[Tensor, "B NV 4 4"],
    intrinsics: Float[Tensor, "B NV 3 3"],
    local_features: Float[Tensor, "B NV C H W"],
    raster_point_radius: float = 0.0075,  # point size
    raster_points_per_pixel: int = 1,  # a single point per pixel, for now
    bin_size: int = 0,
    # ('min', 'max', 'sum', 'mean', 'prod', "fuse_mean_std_cat") or a custom function
    fuse_function: str | Callable[[Tensor, List[int]], Tensor] = "mean",
):
    if fuse_function == "fuse_mean_std_cat":
        fuse_function = fuse_mean_std_cat  # doubles the feature dimension

    points = repeat(points, "B NP xyz -> (B NV) NP xyz", NV=c2ws.shape[1])
    c2ws, ps = pack([c2ws], "* i j")
    intrinsics, ps = pack([intrinsics], "* i j")
    local_features, ps = pack([local_features], "* C H W")

    local_features_proj = points_projection_single_view(
        points, c2ws, intrinsics, local_features, raster_point_radius, raster_points_per_pixel, bin_size
    )
    local_features_proj = unpack(local_features_proj, ps, "* NP C")[0]
    local_features_proj = reduce(local_features_proj, "B NV NP C -> B NP C", fuse_function).contiguous()
    return local_features_proj


def compute_distance_transform(mask: torch.Tensor):
    # Float[Tensor, "B 1 H W"]) -> Float[Tensor, "B 1 H W"]
    image_size = mask.shape[-1]
    distance_transform = torch.stack([
        torch.from_numpy(cv2.distanceTransform(
            (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
        ) / (image_size / 2))
        for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    ]).unsqueeze(1).clip(0, 1).to(mask.device)
    return distance_transform

def fuse_mean_std_cat(x: Tensor, dim: list[int] = [1], keepdim: bool = False):
    # eg. x: (B, N, C) -> (B, C*2)
    # Instant Multi-View Head Capture through Learnable Registration, Bolkart et al., CVPR 2023
    # https://ps.is.mpg.de/uploads_file/attachment/attachment/711/CVPR2023_Multiview_Face_Capture.pdf
    # https://github.com/TimoBolkart/TEMPEH
    mean = x.mean(dim=dim, keepdim=keepdim)
    std = x.std(dim=dim, keepdim=keepdim)
    return torch.cat([mean, std], dim=-1)
