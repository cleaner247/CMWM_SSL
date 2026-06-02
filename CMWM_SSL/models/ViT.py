# File purpose: CMWM_SSL local Vision Transformer implementation for SSL encoders/decoders.
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)



# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, t, n, e = x.shape
        x = x.view(b * t, n, e)
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out.view(b, t, n, e)
        return out


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequences"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., causal=True):
        super().__init__()
        self.dim = dim
        self.num_heads = heads
        self.head_dim = dim_head
        assert self.head_dim * heads == dim, f"embed dim must be divisible by num heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(self, x):
        # x: [B, T, P, E] for temporal attention across time
        B, T, P, E = x.shape

        # normalize
        x_norm = self.norm(x)

        # project to Q, K, V and split into heads: [B, T, P, E] -> [(B*P), H, T, D]
        q = rearrange(self.q_proj(x_norm), 'b t p (h d) -> (b p) h t d', h=self.num_heads)
        k = rearrange(self.k_proj(x_norm), 'b t p (h d) -> (b p) h t d', h=self.num_heads)
        v = rearrange(self.v_proj(x_norm), 'b t p (h d) -> (b p) h t d', h=self.num_heads)

        k_t = k.transpose(-2, -1)  # [(B*P), H, D, T]

        # attention(q, k, v) = softmax(qk^T / sqrt(d)) v
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim)  # [(B*P), H, T, T]

        # causal mask: for each token t, mask out all tokens after t
        if self.causal:
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask, -torch.inf)

        attn_weights = F.softmax(scores, dim=-1)  # [(B*P), H, T, T]
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # [(B*P), H, T, D]
        attn_output = rearrange(attn_output, '(b p) h t d -> b t p (h d)', b=B, p=P)  # [B, T, P, E]

        # out proj
        attn_out = self.out_proj(attn_output)
        attn_out = self.dropout(attn_out)

        # residual connection
        return x + attn_out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., use_temporal=False, causal=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.use_temporal = use_temporal
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            layer_modules = [
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]
            # add temporal attention if enabled
            if use_temporal:
                layer_modules.insert(1, TemporalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal))
            self.layers.append(nn.ModuleList(layer_modules))

    def forward(self, x):
        # x: [B, T, P, E] if temporal, [B, P, E] if not
        if self.use_temporal:
            # spatial -> temporal -> feedforward
            for layer_modules in self.layers:
                spatial_attn, temporal_attn, ff = layer_modules
                x = spatial_attn(x) + x
                x = temporal_attn(x)  # temporal attention already has residual
                x = ff(x) + x
        else:
            # original behavior: spatial -> feedforward
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, image_size, p_in, dim, depth, heads, mlp_dim, dim_in, dim_out, p_out, dim_head=64, dropout=0.,
                 emb_dropout=0., patch_in_out=(True, True), maxlen=1000, use_temporal=False, causal=True):
        super().__init__()
        self.patch_in_out = patch_in_out
        self.use_temporal = use_temporal
        image_height, image_width = pair(image_size)
        p_h_in, p_w_in = pair(p_in)
        p_h_out, p_w_out = pair(p_out)

        assert image_height % p_h_in == 0 and image_width % p_w_in == 0, 'Image dimensions must be divisible by the patch size.'

        if patch_in_out[0]:
            patch_dim_in = dim_in * p_h_in * p_w_in
            num_patches = (image_height // p_h_in) * (image_width // p_w_in)
            self.num_patches = num_patches
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p_h_in, p2=p_w_in),
                nn.LayerNorm(patch_dim_in),
                nn.Identity() if patch_dim_in == dim else nn.Linear(patch_dim_in, dim),
                nn.LayerNorm(dim)
            )
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, maxlen, dim))
            self.proj_in = nn.Linear(dim_in, dim) if dim_in != dim else nn.Identity()
            print(f"Input projection from {dim_in} to {dim}")

        if patch_in_out[1]:
            patch_dim_out = dim_out * p_h_out * p_w_out
            self.from_patch_embedding = nn.Sequential(
                nn.Identity() if patch_dim_out == dim else nn.Linear(dim, patch_dim_out),
                Rearrange('b (h w) (p1 p2 c)-> b c (h p1) (w p2)', h=image_height // p_h_in,
                          w=image_width // p_w_in, p1=p_h_out,
                          p2=p_w_out),
            )
        else:
            self.proj_out = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

        self.dropout = nn.Dropout(emb_dropout)

        # temporal PE setup - use learnable parameters like spatial
        #if use_temporal:
            # Use learnable temporal positional embedding
            #max_temporal_len = 1000  # maximum sequence length
            #self.temporal_pos_embedding = nn.Parameter(torch.randn(1, max_temporal_len, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, use_temporal, causal)

    def forward(self, x):
        # x can be [B, C, H, W] or [B, T, C, H, W] depending on use_temporal
        if self.use_temporal:
            # Handle sequence input [B, T, C, H, W]


            if self.patch_in_out[0]:
                B, T, C, H, W = x.shape
                # Reshape to process all frames: [B*T, C, H, W]
                x = x.view(B * T, C, H, W)
                x = self.to_patch_embedding(x)  # [B*T, P, E]
            else:
                B, T, P, E = x.shape
                x = x.view(B * T, P, E)
                x = self.proj_in(x)

            # Add spatial positional embedding
            spatial_pe = self.pos_embedding[:, :x.size(1), :]  # [1, P, E]
            x = x + spatial_pe

            # Reshape to [B, T, P, E] for temporal processing
            _, P, E = x.shape
            x = x.view(B, T, P, E)

            # Add learnable temporal positional embedding
            #temporal_pe = self.temporal_pos_embedding[:, :T, :]  # [1, T, E]
            #x = x + temporal_pe[:, :, None, :]  # [B, T, P, E]

            x = self.dropout(x)
            x = self.transformer(x)  # [B, T, P, E]

            # Reshape back for output projection
            x = x.view(B * T, P, E)

            if self.patch_in_out[1]:
                x = self.from_patch_embedding(x)  # [B*T, C, H, W]
                _, C_out, H_out, W_out = x.shape
                x = x.view(B, T, C_out, H_out, W_out)
            else:
                x = self.proj_out(x)  # [B*T, P, dim_out]
                x = x.view(B, T, P, -1)
        else:
            # Original single frame processing [B, C, H, W]
            if self.patch_in_out[0]:
                B, T, C, H, W = x.shape

                x = x.view(B * T, C, H, W)
                x = self.to_patch_embedding(x)  # [B*T, P, E]
                _, P, E = x.shape
            else:
                B, T, P, E = x.shape

                x = x.view(B * T, P, E)

                x = self.proj_in(x)
            _, P, E = x.shape

            x += self.pos_embedding[:, :x.size(1), :]
            x = self.dropout(x)
            x = x.view(B, T, P, -1)


            x = self.transformer(x)


            x = x.view(B * T, P, E)


            if self.patch_in_out[1]:
                x = self.from_patch_embedding(x)  # [B*T, C, H, W]
                _, C_out, H_out, W_out = x.shape
                x = x.view(B, T, C_out, H_out, W_out)
            else:
                x = self.proj_out(x)  # [B*T, P, dim_out]
                x = x.view(B, T, P, -1)
                #print(f"Output projection from {E} to {x.shape[-1]}")
        return x