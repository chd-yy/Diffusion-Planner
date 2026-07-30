import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

from hdp_nuplan.model.diffusion_utils.sampling import dpm_sampler
from hdp_nuplan.model.diffusion_utils.sde import SDE, VPSDE_linear
from hdp_nuplan.utils.normalizer import ObservationNormalizer, StateNormalizer
from hdp_nuplan.model.module.mixer import MixerBlock
from hdp_nuplan.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        self.dit = DiT(
            sde=self._sde,
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth,
            output_dim=4, # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type,
            future_length=config.future_len
        )
        
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        
    @property
    def sde(self):
        return self._sde
    
    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """
        # Extract context encoding
        ego_neighbor_encoding = encoder_outputs['encoding']
        B = ego_neighbor_encoding.shape[0]
        route_lanes = inputs['route_lanes']
        ego_v = inputs['ego_current_state'][:, 4:6] # take the v_x v_y of current states for dit embedding

        if self.training:
            sampled_trajectories = inputs['sampled_trajectories'] # [B, V_future, 4]
            diffusion_time = inputs['diffusion_time']
            neighbor_current_mask = None

            return {
                    "score": self.dit(
                        sampled_trajectories, 
                        diffusion_time,
                        ego_neighbor_encoding,
                        route_lanes,
                        ego_v,
                    ).reshape(B, -1, 4)
                }
        else:
            # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            xT = (torch.randn(B, self._future_len, 4).to(ego_neighbor_encoding.device) * 0.1)
            
            x0 = dpm_sampler(
                        self.dit,
                        xT,
                        other_model_params={
                            "cross_c": ego_neighbor_encoding, 
                            "route_lanes": route_lanes,
                            "ego_current_states": ego_v,                           
                        },
                        dpm_solver_params={},
                        model_wrapper_params={},
                )
            x0 = self._state_normalizer.inverse(x0.reshape(B, -1, 4))
            x0 = torch.cat([
                torch.cumsum(x0[..., :2], dim=-2),
                x0[..., 2:]
            ], dim=-1)

            return {
                    "prediction": x0
                }

        
class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        '''
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, -1)


class DiT(nn.Module):
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start", future_length=80):
        super().__init__()
        
        assert model_type in ["noise", "score", "x_start", "v"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(future_length, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.ego_state_proj = nn.Linear(2, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
               
    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, route_lanes, ego_current_states):
        """
        Forward pass of DiT.
        x: (B, T, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        B, T, _ = x.shape
        
        x = self.preproj(x)

        x_embedding = self.agent_embedding.weight[None, :, :].expand(B, -1, -1) # (B, T, D) 
        ego_state_embedding = self.ego_state_proj(ego_current_states)
        ego_state_embedding = ego_state_embedding[:, None, :]
        x = x + x_embedding + ego_state_embedding

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)
        
        for block in self.blocks:
            x = block(x, cross_c, y)
        
        x = self.final_layer(x, y)
        
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start" or self._model_type == "noise" or self._model_type == 'v':
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")