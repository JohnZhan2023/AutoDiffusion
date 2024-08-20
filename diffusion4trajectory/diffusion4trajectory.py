import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from diffusion.transformer import Transformer
from diffusion.wrapper import DiffusionWrapper
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformer4planning.models.encoder.nuplan_raster_encoder import NuplanRasterizeEncoder
from transformer4planning.models.backbone.str_base import STRConfig
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class LTMOutput(CausalLMOutputWithCrossAttentions):
    loss: Optional[torch.FloatTensor] = None
    pred_dict: Optional[Dict[str, torch.FloatTensor]] = None
    logits: Optional[torch.FloatTensor] = None
    loss_items: Optional[torch.FloatTensor] = None


class DiffusionConfig(STRConfig):
    n_cond_steps: int = 1
    cond_dim: int = 128
    map_cond: bool = False
    n_cond_layers: int = 1
    timesteps: int = 1000
    objective: str = "gaussian"
    beta_schedule: str = "linear"
    data_path: str = "data"
    debug: bool = False
    seed: int = 42
    max_eval_samples: int = 100
    max_steps: int = 1000
    warmup_steps: int = 100


class diffusion4trajectory(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.encoder_config = config
        self._build_model()
        
    def _build_model(self):
        transformer = Transformer(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            horizon=self.config.horizon,
            n_cond_steps=self.config.n_cond_steps,
            cond_dim=self.config.cond_dim,
            map_cond=self.config.map_cond,
            n_cond_layers=self.config.n_cond_layers,
        )
        self.diffusion = DiffusionWrapper(
            model=transformer,
            timesteps=self.config.timesteps,
            objective=self.config.objective,
            beta_schedule=self.config.beta_schedule,
        )
        self.encoder = NuplanRasterizeEncoder(self.encoder_config)
    
    def preprocess(self, label):
        # set the first location as the anchor
        first_row_first_three = label[:, 0, :3]
        first_row_first_three = first_row_first_three.unsqueeze(1)
        label[:, :, :3] -= first_row_first_three
        return label
    
    @torch.no_grad()
    def generate(self, **kwargs):
        self.eval()
        input_embeds, info_dict,maps = self.encoder(is_training=self.training, **kwargs)
        raw_trajectory_label = kwargs.get("raw_trajectory_label", None)
        raw_trajectory_label = self.preprocess(raw_trajectory_label)
        
        # make a mask
        mask = torch.ones(maps.shape[0], self.config.horizon, self.config.output_dim).to(maps.device)
        mask[:,:5] = 0
        mask[:,-20:] = 0 

        
        cond = maps
        noise = torch.randn(cond.shape[0], self.config.horizon, self.config.output_dim).to(cond.device)
        
        label = raw_trajectory_label * (1 - mask) + noise * mask
        
        trajectory = self.diffusion(prior=cond, trajectory_label=label, mask = mask)
        pred_dict = {
            "traj_logits": trajectory
        }
        print("trajectory", trajectory[0,::10])
        return pred_dict
        
    def forward(self, **kwargs):
        self.train()
        label = kwargs.get("raw_trajectory_label", None)
        label = self.preprocess(label)
        input_embeds, info_dict, maps = self.encoder(is_training=self.training, **kwargs)
        cond = maps
        loss = self.diffusion(cond, label)
        pred_dict = {
            "traj_logits": label
        }
        
        output_dict = LTMOutput(
            loss=loss,
            loss_items=loss,
            pred_dict=pred_dict,
            logits=label
            
        )
        return output_dict