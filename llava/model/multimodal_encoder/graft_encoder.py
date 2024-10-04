import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
from torchvision import transforms
from PIL import Image

CKPT = "../../../multimodal_representation_learning/outputs/2024-08-05-14-18-55_CLIP_Contrastive_M2O_Cross_Image_Only_wd_1e-2_lr_1e-5_fixed_normalization_fixed_projector_initialization/checkpoints/last.ckpt"
# ckpt = "../multimodal_representation_learning/outputs/2024-08-06-13-08-56_CLIP_Contrastive_M2O_Cross_Image_Only_wd_1e-2_lr_1e-5_fixed_normalization_fixed_projector_initialization_336/checkpoints/last.ckpt"

class ModelOutput:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states

class GRAFT(nn.Module):
    def __init__(self, CLIP_version="openai/clip-vit-base-patch16", temp=False, bias_projector=True):
        super().__init__()
        # satellite image backbone
        self.satellite_image_backbone = CLIPVisionModelWithProjection.from_pretrained(CLIP_version).to("cpu")
        self.patch_size = self.satellite_image_backbone.config.patch_size
        self.image_processor = CLIPImageProcessor.from_pretrained(CLIP_version)
        
        # print(CLIP_version)
        # for name, param in self.satellite_image_backbone.named_parameters():
        #     print(f"Layer: {name}, Shape: {param.shape}")


        self.projector = nn.Sequential(
            nn.LayerNorm(self.satellite_image_backbone.config.hidden_size, eps=self.satellite_image_backbone.config.layer_norm_eps),
            nn.Linear(self.satellite_image_backbone.config.hidden_size, self.satellite_image_backbone.config.projection_dim, bias=bias_projector),
        )
        
        self.patch_size = self.satellite_image_backbone.config.patch_size
        self.norm_dim = -1

        self.temp = temp
        if temp:
            self.register_buffer("logit_scale", torch.ones([]) * (1 / 0.07))
    
    def forward(self, image_tensor, output_hidden_states=False):
        # Extract features from satellite images
        # B x 197 x 768 for VIT-B/16
        # print("image_tensor", image_tensor.shape)
        output = self.satellite_image_backbone(image_tensor, output_hidden_states=output_hidden_states)
        hidden_state = output.last_hidden_state
        # B x 197 x 512
        # print("hidden_state", hidden_state.shape)
        satellite_image_features = F.normalize(self.projector(hidden_state), dim=self.norm_dim)
        # print("satellite_image_features", satellite_image_features.shape)
        # Return as a custom object
        return ModelOutput(satellite_image_features, output.hidden_states) if output_hidden_states else ModelOutput(satellite_image_features, None)

    def forward_features(self, image_tensor):
        # Extract features from satellite images
        # B x 512 for VIT-B/16
        embed = self.satellite_image_backbone(image_tensor).image_embeds
        # B x 512
        satellite_image_features = F.normalize(embed)
        return satellite_image_features

class GRAFTVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # self.image_processor = GRAFTImageProcessor(image_size=(224, 224))
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        self.vision_tower = GRAFT(temp=True, bias_projector=False).to("cuda")
        # for name, param in self.vision_tower.named_parameters():
        #     print(f"Layer: {name}, Shape: {param.shape}")
        self.image_processor = self.vision_tower.image_processor

        if CKPT:
            print("CKPT", CKPT)
            sd = torch.load(CKPT)
            print(sd['state_dict'].keys())
            self.vision_tower.load_state_dict(sd['state_dict'], strict=True)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # print("hidden states",image_forward_outs.hidden_states.shape)
        # print("image_features", image_features.shape)
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        # print("image_features", image_features.shape)
        return image_features
    
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        # print("forward image_features", image_features.shape)
        return image_features

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.satellite_image_backbone.config
        else:
            return self.cfg_only

class GRAFTVisionTowerS2(GRAFTVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        
        # self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = getattr(args, 's2_scales', '224,448,672')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]
        
        super().__init__(vision_tower, args, delay_load)
        
        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            # self.image_processor = GRAFTImageProcessor(image_size=(self.s2_image_size, self.s2_image_size))
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size
            
    # def load_model(self, device_map=None):
    #     if self.is_loaded:
    #         return

    #     # self.image_processor = GRAFTImageProcessor(image_size=(self.s2_image_size, self.s2_image_size))
    #     self.vision_tower = GRAFT(temp=True, bias_projector=False).to("cuda")
    #     for name, param in self.vision_tower.named_parameters():
    #         print(f"Layer: {name}, Shape: {param.shape}")
    #     self.image_processor = self.vision_tower.image_processor

    #     if CKPT:
    #         print("CKPT", CKPT)
    #         sd = torch.load(CKPT)
    #         self.vision_tower.load_state_dict(sd['state_dict'], strict=True)

    #     self.vision_tower.requires_grad_(False)
    #     self.is_loaded = True
    
    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features
    
    
    @torch.no_grad()
    def forward(self, images):
        print("images", images.shape)
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
    