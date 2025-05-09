import os
import torch
import torch.nn as nn
import requests
from tqdm import tqdm
from dinov2.models.vision_transformer import vit_small
import numpy as np
import einops
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import torchvision
from typing import Callable
from torch import Tensor
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module

class GroupNormResNet18(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = "ResNet18_Weights.IMAGENET1K_V1" if pretrained else None
        # Set up backbone.
        backbone_model = getattr(torchvision.models, "resnet18")(
            weights=weights,
        )
        if pretrained:
            print(f"WARNING {weights} pretraining with group norm will mess up the weights")
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        self.backbone = _replace_submodules(
            root_module=self.backbone,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
        )

    @property
    def out_dim(self):
        return 512

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    

class PretrainedResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone_model = getattr(torchvision.models, "resnet18")(
            weights="ResNet18_Weights.IMAGENET1K_V1" if pretrained else None,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(self.backbone_model, return_layers={"layer4": "feature_map"})

    @property
    def out_dim(self):
        return self.backbone_model.fc.in_features
    
    def forward(self, x):
        return self.backbone(x)["feature_map"]

def load_pretrained_weights(model, model_name, download_url, weight_dir=os.path.join(os.path.dirname(__file__), '../../../models')):
    # Ensure the directory exists
    os.makedirs(weight_dir, exist_ok=True)
    
    # Define the path where weights will be saved
    weight_path = os.path.join(weight_dir, f"{model_name}.pth")

    # Download weights if they don't exist
    if not os.path.exists(weight_path):
        print(f"Downloading pretrained weights for {model_name}...")

        # Stream the download with a progress bar
        response = requests.get(download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        # Write the file in chunks and display progress
        with open(weight_path, 'wb') as f, tqdm(
            desc=f"Downloading {model_name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"Downloaded and saved to {weight_path}")

    # Load weights into the model
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained weights from {weight_path}")

    return model

class DINOv2(nn.Module):
    def __init__(self, ret_attn_layers=[]):
        super().__init__()
        # self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

        self.backbone = vit_small(
            patch_size=14,
            img_size=526,
            init_values=1.0,
            num_register_tokens=4,
            block_chunks=0,
            ret_attn_layers=ret_attn_layers,
        )

        # Load the pretrained weights.
        model_name = "dinov2_vits14_reg4_pretrain"
        download_url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth'
        self.backbone = load_pretrained_weights(self.backbone, model_name, download_url)

    @property
    def out_dim(self):
        return 384
    
    @property
    def patch_size(self):
        return 14
    
    @property
    def embed_dim(self):
        return 384
    
    def forward(self, x):
        w = x.shape[-1]
        assert w % self.patch_size == 0, f"Image width {w} must be divisible by patch size {self.patch_size}"
        patch_w = w // self.patch_size
        out = self.backbone.forward_features(x, output_attentions=False)
        features = einops.rearrange(out['x_norm_patchtokens'], 'b (h w) d -> b d h w',  w=patch_w)
        return features

    def forward_features(self, x, output_attentions=True):
        return self.backbone.forward_features(x, output_attentions=output_attentions)



class GroupNormResNet18UNet(nn.Module):
    def __init__(self, in_channels, classes=1):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",    
            in_channels=in_channels,                 
            classes=classes,                  
        )
        self.unet = _replace_submodules(
            root_module=self.unet,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
        )

    def forward(self, x):
        encoder_out = self.unet.encoder(x)
        features = encoder_out[-1]
        masks = self.unet.segmentation_head(self.unet.decoder(encoder_out))
        return masks, features