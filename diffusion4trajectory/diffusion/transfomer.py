from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from transformer4planning.models.diffusion_loss.positional_embedding import SinusoidalPosEmb
# Credit to Cheng Chi's implementation of TransformerForDiffusion (https://github.com/real-stanford/diffusion_policy/tree/main)

logger = logging.getLogger(__name__)

class Transformer(nn.Module):
    def __init__(self,
            input_dim: int, 
            output_dim: int,
            horizon: int,

            n_layer: int = 4,
            n_head: int = 8,
            n_emb: int = 32,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder   
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1

        
        # input embedding stem
        self.out_features = output_dim
        self.input_emb = nn.Linear(input_dim, n_emb) # 4->n_emb
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        
        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.Mish(),
            nn.Linear(4 * n_emb, n_emb)
        )
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )


        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only


    
    def forward(self,
        noisy_x: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        **kwargs):
        """
        noisy_x: (B,T,D)
        timestep: (B,)
        
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=noisy_x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(noisy_x.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps.expand(noisy_x.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(noisy_x)

        cond_embeddings = time_emb

   
            
        tc = cond_embeddings.shape[1] #791
        position_embeddings = self.cond_pos_emb[
            :, :tc, :
        ]  # each position maps to a (learnable) 

        x = self.drop(cond_embeddings + position_embeddings)
        x = self.encoder(x)
        memory = x
        # (B,T_cond,n_emb)
        # decoder
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # (B,T,n_emb)
        
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask
        )
        # (B,T,n_emb)
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x

def main():
    transformer = Transformer(4, 4, 10)

if __name__ == "__main__":
    main()