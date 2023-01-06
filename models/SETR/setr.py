import torch
import numpy as np
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


    def forward(self, x):
        x = self.proj(
            x
        )
        
        x = x.flatten(2)
        x = x.transpose(1,2)

        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0, proj_p=0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        dp = (
            q @ k_t
        ) * self.scale

        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(
            1,2
        )
        weighted_avg = weighted_avg.flatten(2)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):

        x = self.fc1(
            x
        )
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x 



class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, qkv_bias=True, p=0., attn_p=0.) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, 
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features = dim, 
            hidden_features=hidden_features,
            out_features=dim
        )


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling) 

class ViT(nn.Module):
    def __init__(self,img_size=384, patch_size=16, in_chans=3, n_classes=1, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4, qkv_bias=True, p=0., attn_p=0.) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim)) #(1, 576, 768)
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim = embed_dim,
                    n_heads = n_heads, 
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p

                )
                for _ in range(depth)
            ]
        )

        # self.seg_head = SegmentationHead(in_chans, embed_dim, kernel_size=1)

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.decoder = Decoder(in_channels=embed_dim, out_channels=n_classes)

    def forward(self, x):

        n_samples = x.shape[0]
        x = self.patch_embed(x) # (1, 768, 24, 24)
        x = x + self.pos_embed
        x = self.pos_drop(x) # (1, 576, 768)


        for block in self.blocks:
            x = block(x)


        x = self.norm(x)

        x = self.decoder(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels=in_channels
        self.decoder = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                ),
                nn.Upsample(scale_factor=(384 / 26), mode="bilinear", align_corners=True)
            )
    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            24,
            24,
            self.in_channels,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


    def forward(self, x):
        x = self._reshape_output(x)
        x = self.decoder(x)
        return x


if __name__=='__main__':

    model = ViT()
    dummy = torch.randn(10, 3, 384, 384)
    print(model(dummy).shape)
