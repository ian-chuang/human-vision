
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from lerobot.common.policies.utils import get_output_shape
import einops
from dataclasses import dataclass, field
import segmentation_models_pytorch as smp
import numpy as np
from torchvision.ops import roi_align
import math
from lerobot.common.policies.vision_encoders import GroupNormResNet18, GroupNormResNet18UNet

@dataclass
class GazeVisionEncoderConfig:
    resize_shape: tuple[int, int] = (240, 320)
    crop_shape: tuple[int, int] = (240, 320)
    crop_is_random: bool = True
    use_spatial_softmax: bool = True
    num_kp: int = 32
    out_dim: int = 64
    dropout: float = 0.1

    use_gaze: bool = False
    gaze_crop_shape: tuple[int, int] = (48, 64)
    gaze_add_small_features: bool = False
    gaze_peripheral_dropout: float = 0.3

    eye_keys: list[str] = field(default_factory=lambda: ["left_eye"])
    image_keys: list[str] = field(default_factory=lambda: ["observation.images.zed_cam_left"])

class GazeVisionEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: GazeVisionEncoderConfig):
        super().__init__()
        self.config = config
        
        self.backbone = GroupNormResNet18(pretrained=True)
        self.backbone_out_dim = self.backbone.out_dim
        dummy_shape = (1, 3, *config.gaze_crop_shape) if config.use_gaze else (1, 3, *config.crop_shape)
        self.feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        print(f"Feature map shape: {self.feature_map_shape}")
        if self.config.use_spatial_softmax:
            self.pool = SpatialSoftmax(self.feature_map_shape, num_kp=config.num_kp)
            self.linear = nn.Linear(config.num_kp * 2, config.out_dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(self.backbone_out_dim, config.out_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)   

        if config.use_gaze:
            self.gaze_model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights="imagenet",    
                in_channels=3,                 
                classes=1,                  
            )
            if config.gaze_add_small_features:
                self.small_backbone = GroupNormResNet18(pretrained=True)
                self.small_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.small_linear = nn.Linear(self.backbone_out_dim, config.out_dim)
                self.small_gelu = nn.GELU()
                self.small_dropout = nn.Dropout(config.gaze_peripheral_dropout)

            self.spatial_softmax = SpatialSoftmax((1, *self.config.gaze_crop_shape))

    def forward(self, batch: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        ret = {}
        ret['loss'] = 0.0

        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        
        images = torch.stack(
            [batch[key] for key in self.config.image_keys], dim=2
        )
        batch_size, n_obs_steps = images.shape[:2]
        images = einops.rearrange(images, 'b s n c h w -> (b s n) c h w')

        # resize
        if self.config.resize_shape != images.shape[-2:]:
            if self.training: print("WARNING you are resizing img while training will be slow stupid")
            images = torchvision.transforms.Resize(self.config.resize_shape)(images)

        # crop
        if self.config.crop_shape != images.shape[-2:]:
            if self.training and self.config.crop_is_random:
                crop_indices = torchvision.transforms.RandomCrop.get_params(
                    images, output_size=self.config.crop_shape
                )
                i, j, h, w = crop_indices  
            else:
                i = (self.config.resize_shape[0] - self.config.crop_shape[0]) // 2
                j = (self.config.resize_shape[1] - self.config.crop_shape[1]) // 2
                h = self.config.crop_shape[0]
                w = self.config.crop_shape[1]

            images = torchvision.transforms.functional.crop(images, i, j, h, w)
        else:
            i, j, h, w = 0, 0, images.shape[-2], images.shape[-1]

        ret['images'] = images

        # adjust eyes according to crop
        if self.training and len(self.config.eye_keys) > 0:
            eyes = torch.stack(
                [batch[key] for key in self.config.eye_keys], dim=2
            )
            assert eyes.shape[3] == 2, "Expected 2 channels in eyes"
            eyes = einops.rearrange(eyes, 'b s n d -> (b s n) d')
            eyes[..., 0] = (((eyes[..., 0]+1)/2) * self.config.resize_shape[1] - j) / w * 2 - 1
            eyes[..., 1] = (((eyes[..., 1]+1)/2) * self.config.resize_shape[0] - i) / h * 2 - 1
            ret['eyes'] = eyes

        if self.config.use_gaze:
            small_images = torch.nn.functional.interpolate(
                images, size=self.config.gaze_crop_shape, mode='bilinear', align_corners=False
            )
            gaze_heatmap = self.gaze_model(small_images)
            ret['gaze_features'] = gaze_heatmap
            pred_eyes = self.spatial_softmax(gaze_heatmap).squeeze(1)

            if self.training and len(self.config.eye_keys) > 0:
                loss = F.mse_loss(
                    pred_eyes,
                    eyes,
                    reduction='mean'
                )
                ret['gaze_loss'] = loss.item()
                ret['loss'] += loss

            boxes = torch.zeros(pred_eyes.shape[0], 5, device=pred_eyes.device)
            h, w = images.shape[-2:]
            new_h, new_w = small_images.shape[-2:]
            eye_pixel_x = ((pred_eyes[:, 0] + 1) / 2) * w
            eye_pixel_y = ((pred_eyes[:, 1] + 1) / 2) * h
            boxes[:, 0] = torch.arange(pred_eyes.shape[0], device=pred_eyes.device)
            boxes[:, 1] = eye_pixel_x - new_w / 2
            boxes[:, 2] = eye_pixel_y - new_h / 2
            boxes[:, 3] = eye_pixel_x + new_w / 2
            boxes[:, 4] = eye_pixel_y + new_h / 2     

            images = roi_align(
                images,
                boxes,
                output_size=small_images.shape[-2:],
            )   

            ret['small_images'] = small_images
            ret['cropped_images'] = images
            ret['pred_eyes'] = einops.rearrange(pred_eyes, '(b s n) k -> b s n k', b=batch_size, s=n_obs_steps)

        # forward through the backbone
        features = self.backbone(images)
        features = torch.flatten(self.pool(features), start_dim=1)
        features = self.gelu(self.linear(features))
        features = self.dropout(features)

        if self.config.use_gaze:
            features = torch.concat([features, pred_eyes], dim=1)

        if self.config.use_gaze and self.config.gaze_add_small_features:
            small_features = self.small_backbone(small_images)
            small_features = torch.flatten(self.small_pool(small_features), start_dim=1)
            small_features = self.gelu(self.small_linear(small_features))
            small_features = self.small_dropout(small_features)
            features = torch.concat([features, small_features], dim=1)

        features = einops.rearrange(features, '(b s n) d -> b s n d', b=batch_size, s=n_obs_steps)

        return features, ret

class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints