"""Untrained set-attention routing head. Not a deployable flight agent.

No positional/ID embeddings: permuting observed balloon rows permutes logits.
Action N is always DEFER_TO_VALIDATED_PLANNER, not an unsafe zero-thrust command.
Before deployment a caller must supply feasible candidates, train weights and
validate the proposed full route. This module cannot bypass physical checks.
"""
import math
import torch
from torch import nn
from BalloonPoppingGymEnv.agents.route_attention_features import NODE_FEATURES, CONTEXT_FEATURES


class RouteAttentionScorer(nn.Module):
    def __init__(self, width=128, heads=4, layers=2):
        super().__init__()
        if width < 1 or heads < 1 or layers < 1 or width % heads:
            raise ValueError('Positive architecture sizes; width must be divisible by heads')
        self.node_encoder = nn.Linear(NODE_FEATURES,width)
        self.context_encoder = nn.Linear(CONTEXT_FEATURES,width)
        layer = nn.TransformerEncoderLayer(width,heads,dim_feedforward=4*width,
                                           dropout=0.,batch_first=True,activation='gelu')
        self.encoder = nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.query = nn.Linear(width,width)
        self.key = nn.Linear(width,width)
        self.fallback = nn.Linear(width,1)
        self.value = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.width = width

    def forward(self, nodes, context, visible_mask, candidate_mask):
        if (nodes.ndim != 3 or nodes.shape[-1] != NODE_FEATURES
                or context.shape != (nodes.shape[0],CONTEXT_FEATURES)
                or visible_mask.shape != nodes.shape[:2]
                or candidate_mask.shape != nodes.shape[:2]
                or visible_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool):
            raise ValueError('Invalid batched routing feature shapes/masks')
        if torch.any(candidate_mask & ~visible_mask):
            raise ValueError('An invisible balloon cannot be a candidate')
        # Discard padding before projection so NaNs from unavailable rows cannot
        # leak through attention. Real visible features must always be finite.
        nodes = torch.where(visible_mask.unsqueeze(-1),nodes,torch.zeros_like(nodes))
        if not torch.isfinite(nodes).all() or not torch.isfinite(context).all():
            raise ValueError('Visible routing features must be finite')
        tokens = torch.cat([self.context_encoder(context).unsqueeze(1),self.node_encoder(nodes)],dim=1)
        keep = torch.cat([torch.ones((nodes.shape[0],1),dtype=torch.bool,device=nodes.device),visible_mask],dim=1)
        encoded = self.encoder(tokens,src_key_padding_mask=~keep)
        summary = encoded[:,0]
        logits = torch.einsum('bd,bnd->bn',self.query(summary),self.key(encoded[:,1:]))/math.sqrt(self.width)
        logits = logits.masked_fill(~candidate_mask,-torch.inf)
        # A real context token avoids all-masked attention, including N=0.
        logits = torch.cat([logits,self.fallback(summary)],dim=1)
        return logits,self.value(summary).squeeze(-1)
