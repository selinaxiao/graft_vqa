# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
# from torchvision import transforms
# from PIL import Image

# class GRAFTImageProcessor:
#     def __init__(self, image_size=(224, 224)):
#         self.image_size = image_size
#         print("image processor init image_size",image_size)
#         self.transform = transforms.Compose([
#             transforms.Resize(image_size),  # Resize to the target size
#             transforms.CenterCrop(image_size),  # Ensure consistent size
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                                  std=[0.26862954, 0.26130258, 0.27577711])
#         ])
#         # self.transform = transforms.Compose([transforms.Resize((224, 224)),
#         # transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

#     def __call__(self, images):
#         if isinstance(images, list):
#             return {"pixel_values": torch.stack([self.process_image(image) for image in images])}
#         elif isinstance(images, torch.Tensor) and images.dim() == 4:
#             return {"pixel_values": self.process_image(images)}
#         else:
#             return {"pixel_values": self.process_image(images)}

#     def process_image(self, image):
#         if isinstance(image, torch.Tensor):
#             if image.dim() == 4:  # Batch of images
#                 return torch.stack([self.process_single_image(img) for img in image])
#             else:  # Single image
#                 return self.process_single_image(image)
#         elif isinstance(image, (str, Image.Image)):
#             return self.process_single_image(image)

#     def process_single_image(self, image):
#         if isinstance(image, torch.Tensor):
#             if image.dim() == 3:  # Single image tensor
#                 image = transforms.ToPILImage()(image)
#         if isinstance(image, str):
#             image = Image.open(image)
#         elif isinstance(image, Image.Image):
#             image = image
#         # print("image",image.size)
#         # print("after transform", self.transform(image).shape)
#         # Apply the transformations: resize, crop, and normalize
#         return self.transform(image)

#     def preprocess(self, images, return_tensors='pt'):
#         """
#         Preprocesses images to the required format.
        
#         Args:
#             images: A single image or a list of images.
#             return_tensors: The format to return the tensors in. Defaults to 'pt' (PyTorch tensors).
        
#         Returns:
#             A dictionary containing the processed pixel values.
#         """
#         if isinstance(images, list):
#             processed_images = [self.process_image(image) for image in images]
#         else:
#             processed_images = [self.process_image(images)]

#         if return_tensors == 'pt':
#             pixel_values = torch.stack(processed_images)
#         else:
#             raise ValueError(f"Unsupported return_tensors value: {return_tensors}")

#         return {"pixel_values": pixel_values}

# class GRAFT(nn.Module):
#     def __init__(self, CLIP_version="openai/clip-vit-base-patch16", temp=False, bias_projector=True):
#         super().__init__()
#         # satellite image backbone
#         self.satellite_image_backbone = CLIPVisionModelWithProjection.from_pretrained(CLIP_version)
#         self.patch_size = self.satellite_image_backbone.config.patch_size
#         self.image_processor = CLIPImageProcessor.from_pretrained(CLIP_version)

#         self.projector = nn.Sequential(
#             nn.LayerNorm(self.satellite_image_backbone.config.hidden_size, eps=self.satellite_image_backbone.config.layer_norm_eps),
#             nn.Linear(self.satellite_image_backbone.config.hidden_size, self.satellite_image_backbone.config.projection_dim, bias=bias_projector),
#         )
#         self.patch_size = self.satellite_image_backbone.config.patch_size
#         self.norm_dim = -1

#         self.temp = temp
#         if temp:
#             self.register_buffer("logit_scale", torch.ones([]) * (1 / 0.07))

#     def forward(self, image_tensor):
#         # Extract features from satellite images
#         # B x 197 x 768 for VIT-B/16
#         print("image_tensor",image_tensor.shape)
#         hidden_state = self.satellite_image_backbone(image_tensor).last_hidden_state
#         # B x 197 x 512
#         satellite_image_features = F.normalize(self.projector(hidden_state), dim=self.norm_dim)
#         # get the satellite image features
#         return satellite_image_features

#     def forward_features(self, image_tensor):
#         # Extract features from satellite images
#         # B x 512 for VIT-B/16
#         embed = self.satellite_image_backbone(image_tensor).image_embeds
#         # B x 512
#         satellite_image_features = F.normalize(embed)
#         return satellite_image_features

# class GRAFTVisionTower(nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False, ckpt_path=None):
#         super().__init__()
#         self.is_loaded = False
#         self.vision_tower_name = vision_tower
#         self.ckpt_path = ckpt_path
#         # self.image_processor = GRAFTImageProcessor(image_size=(224, 224))
        
#         if not delay_load:
#             self.load_model()
#         elif getattr(args, 'unfreeze_mm_vision_tower', False):
#             self.load_model()
#         else:
#             self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             return

#         self.vision_tower = GRAFT(temp=True, bias_projector=False).to("cuda")
#         self.image_processor = vision_tower.image_processor

#         if self.ckpt_path:
#             sd = torch.load(self.ckpt_path)
#             self.vision_tower.load_state_dict(sd['state_dict'], strict=True)

#         self.vision_tower.requires_grad_(False)
#         self.is_loaded = True

#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
#         if self.select_feature == 'patch':
#             image_features = image_features[:, 1:]
#         elif self.select_feature == 'cls_patch':
#             image_features = image_features
#         else:
#             raise ValueError(f'Unexpected select feature: {self.select_feature}')
#         return image_features
    
#     @torch.no_grad()
#     def forward_feature(self, images):
#         if images.dtype == torch.bfloat16:
#             images = images.to(torch.float32)
#         if torch.min(images) < 0 or torch.max(images) > 1:
#             images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
            
#         if isinstance(images, list):
#             images = torch.stack([self.image_processor(image)["pixel_values"] for image in images])
#         else:
#             images = self.image_processor(images)["pixel_values"]

#         # Ensure `images` is a tensor and move to the correct device and dtype
#         resized_images = images.to(device=self.device, dtype=self.dtype)
#         print("resized images", resized_images.shape)

#         # Forward pass through vision tower
#         image_forward_outs = self.vision_tower(resized_images)
#         return image_forward_outs

#     @torch.no_grad()
#     def forward(self, images):
#         image_features = self.forward_feature(images)
#         return image_features

#     @property
#     def dtype(self):
#         return next(self.vision_tower.parameters()).dtype

#     @property
#     def device(self):
#         return next(self.vision_tower.parameters()).device

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.satellite_image_backbone.config
#         else:
#             return self.cfg_only

# class GRAFTVisionTowerS2(GRAFTVisionTower):
#     def __init__(self, vision_tower, args, delay_load=False, ckpt_path=None):
        
#         self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
#         # self.s2_scales = getattr(args, 's2_scales', '224,224,224')
#         self.s2_scales = list(map(int, self.s2_scales.split(',')))
#         self.s2_scales.sort()
#         self.s2_split_size = self.s2_scales[0]
#         self.s2_image_size = self.s2_scales[-1]
        
#         super().__init__(vision_tower, args, delay_load, ckpt_path)
        
#         try:
#             from s2wrapper import forward as multiscale_forward
#         except ImportError:
#             raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
#         self.multiscale_forward = multiscale_forward

#         if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
#             # self.image_processor = GRAFTImageProcessor(image_size=(self.s2_image_size, self.s2_image_size))
#             self.image_processor.size['shortest_edge'] = self.s2_image_size
#             self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size
            
#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             return

#         # self.image_processor = GRAFTImageProcessor(image_size=(self.s2_image_size, self.s2_image_size))
#         self.vision_tower = GRAFT(temp=True, bias_projector=False).to("cuda")
#         self.image_processor = self.vision_tower.image_processor

#         if self.ckpt_path:
#             sd = torch.load(self.ckpt_path)
#             self.vision_tower.load_state_dict(sd['state_dict'], strict=True)

#         self.vision_tower.requires_grad_(False)
#         self.is_loaded = True

#     # @torch.no_grad()
#     # def forward_feature(self, images):
#     #     image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
#     #     image_features = self.feature_select(image_forward_outs).to(images.dtype)
#     #     return image_features
    
    
#     @torch.no_grad()
#     def forward(self, images):
#         if images.dtype == torch.bfloat16:
#             images = images.to(torch.float32)
#         image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
#         # image_features = self.forward_feature(images)
#         return image_features

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size * len(self.s2_scales)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
from torchvision import transforms
from PIL import Image

# class GRAFTImageProcessor:
#     def __init__(self, image_size=(224, 224)):
#         self.image_size = image_size
#         print("image processor init image_size",image_size)
#         self.transform = transforms.Compose([
#             transforms.Resize(image_size),  # Resize to the target size
#             transforms.CenterCrop(image_size),  # Ensure consistent size
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                                  std=[0.26862954, 0.26130258, 0.27577711])
#         ])
#         # self.transform = transforms.Compose([transforms.Resize((224, 224)),
#         # transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

#     def __call__(self, images):
#         if isinstance(images, list):
#             return {"pixel_values": torch.stack([self.process_image(image) for image in images])}
#         elif isinstance(images, torch.Tensor) and images.dim() == 4:
#             return {"pixel_values": self.process_image(images)}
#         else:
#             return {"pixel_values": self.process_image(images)}

#     def process_image(self, image):
#         if isinstance(image, torch.Tensor):
#             if image.dim() == 4:  # Batch of images
#                 return torch.stack([self.process_single_image(img) for img in image])
#             else:  # Single image
#                 return self.process_single_image(image)
#         elif isinstance(image, (str, Image.Image)):
#             return self.process_single_image(image)

#     def process_single_image(self, image):
#         if isinstance(image, torch.Tensor):
#             if image.dim() == 3:  # Single image tensor
#                 image = transforms.ToPILImage()(image)
#         if isinstance(image, str):
#             image = Image.open(image)
#         elif isinstance(image, Image.Image):
#             image = image
#         # print("image",image.size)
#         # print("after transform", self.transform(image).shape)
#         # Apply the transformations: resize, crop, and normalize
#         return self.transform(image)

#     def preprocess(self, images, return_tensors='pt'):
#         """
#         Preprocesses images to the required format.
        
#         Args:
#             images: A single image or a list of images.
#             return_tensors: The format to return the tensors in. Defaults to 'pt' (PyTorch tensors).
        
#         Returns:
#             A dictionary containing the processed pixel values.
#         """
#         if isinstance(images, list):
#             processed_images = [self.process_image(image) for image in images]
#         else:
#             processed_images = [self.process_image(images)]

#         if return_tensors == 'pt':
#             pixel_values = torch.stack(processed_images)
#         else:
#             raise ValueError(f"Unsupported return_tensors value: {return_tensors}")

#         return {"pixel_values": pixel_values}

class GRAFT(nn.Module):
    def __init__(self, CLIP_version="openai/clip-vit-base-patch16", temp=False, bias_projector=True):
        super().__init__()
        # satellite image backbone
        self.satellite_image_backbone = CLIPVisionModelWithProjection.from_pretrained(CLIP_version)
        self.patch_size = self.satellite_image_backbone.config.patch_size
        self.image_processor = CLIPImageProcessor.from_pretrained(CLIP_version)

        self.projector = nn.Sequential(
            nn.LayerNorm(self.satellite_image_backbone.config.hidden_size, eps=self.satellite_image_backbone.config.layer_norm_eps),
            nn.Linear(self.satellite_image_backbone.config.hidden_size, self.satellite_image_backbone.config.projection_dim, bias=bias_projector),
        )
        self.patch_size = self.satellite_image_backbone.config.patch_size
        self.norm_dim = -1

        self.temp = temp
        if temp:
            self.register_buffer("logit_scale", torch.ones([]) * (1 / 0.07))

    def forward(self, image_tensor):
        # Extract features from satellite images
        # B x 197 x 768 for VIT-B/16
        print("image_tensor",image_tensor.shape)
        hidden_state = self.satellite_image_backbone(image_tensor).last_hidden_state
        # B x 197 x 512
        satellite_image_features = F.normalize(self.projector(hidden_state), dim=self.norm_dim)
        # get the satellite image features
        return satellite_image_features

    def forward_features(self, image_tensor):
        # Extract features from satellite images
        # B x 512 for VIT-B/16
        embed = self.satellite_image_backbone(image_tensor).image_embeds
        # B x 512
        satellite_image_features = F.normalize(embed)
        return satellite_image_features

class GRAFTVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, ckpt_path=None):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.ckpt_path = ckpt_path
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
        self.image_processor = vision_tower.image_processor

        if self.ckpt_path:
            sd = torch.load(self.ckpt_path)
            self.vision_tower.load_state_dict(sd['state_dict'], strict=True)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    # @torch.no_grad()
    # def forward_feature(self, images):
    #     if images.dtype == torch.bfloat16:
    #         images = images.to(torch.float32)
    #     if torch.min(images) < 0 or torch.max(images) > 1:
    #         images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
            
    #     if isinstance(images, list):
    #         images = torch.stack([self.image_processor(image)["pixel_values"] for image in images])
    #     else:
    #         images = self.image_processor(images)["pixel_values"]

    #     # Ensure `images` is a tensor and move to the correct device and dtype
    #     resized_images = images.to(device=self.device, dtype=self.dtype)
    #     print("resized images", resized_images.shape)

    #     # Forward pass through vision tower
    #     image_forward_outs = self.vision_tower(resized_images)
    #     return image_forward_outs

    # @torch.no_grad()
    # def forward(self, images):
    #     image_features = self.forward_feature(images)
    #     return image_features
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

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class GRAFTVisionTowerS2(GRAFTVisionTower):
    def __init__(self, vision_tower, args, delay_load=False, ckpt_path=None):
        
        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        # self.s2_scales = getattr(args, 's2_scales', '224,224,224')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]
        
        super().__init__(vision_tower, args, delay_load, ckpt_path)
        
        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            # self.image_processor = GRAFTImageProcessor(image_size=(self.s2_image_size, self.s2_image_size))
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size
            
    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        # self.image_processor = GRAFTImageProcessor(image_size=(self.s2_image_size, self.s2_image_size))
        self.vision_tower = GRAFT(temp=True, bias_projector=False).to("cuda")
        self.image_processor = self.vision_tower.image_processor

        if self.ckpt_path:
            sd = torch.load(self.ckpt_path)
            self.vision_tower.load_state_dict(sd['state_dict'], strict=True)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features
    
    
    # @torch.no_grad()
    # def forward(self, images):
    #     if images.dtype == torch.bfloat16:
    #         images = images.to(torch.float32)
    #     image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
    #     # image_features = self.forward_feature(images)
    #     return image_features
    @torch.no_grad()
    def forward(self, images):
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