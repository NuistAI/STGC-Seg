import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Original Medical-SAM3 bottleneck adapter.

    ``forward_branch`` exposes only the adapter residual.  ``forward`` keeps the
    historical skip-connect behaviour for callers outside the ViT blocks.
    """

    def __init__(
        self,
        D_features,
        mlp_ratio=0.25,
        act_layer=nn.GELU,
        skip_connect=True,
    ):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.bottleneck_dim = D_hidden_features
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward_branch(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        return self.D_fc2(xs)

    def forward(self, x):
        xs = self.forward_branch(x)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class SpatialAdapter(nn.Module):
    """Bottleneck adapter with a local 3x3 depthwise convolution.

    Medical-SAM3's ViT keeps spatial features in ``[B, H, W, C]`` form.  The
    explicit checks below make an accidental ``[B, N, C]`` call fail loudly.
    """

    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.bottleneck_dim = D_hidden_features
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.depthwise_conv = nn.Conv2d(
            D_hidden_features,
            D_hidden_features,
            kernel_size=3,
            padding=1,
            groups=D_hidden_features,
        )
        nn.init.zeros_(self.depthwise_conv.weight)
        self.depthwise_conv.weight.data[:, 0, 1, 1] = 1.0
        nn.init.zeros_(self.depthwise_conv.bias)
        self.act = act_layer()
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward_branch(self, x):
        if x.ndim != 4:
            raise ValueError(
                "SpatialAdapter expects Medical-SAM3 BHWC features, "
                f"received shape={tuple(x.shape)}"
            )
        xs = self.D_fc1(x)
        xs = xs.permute(0, 3, 1, 2).contiguous()
        xs = self.depthwise_conv(xs)
        xs = xs.permute(0, 2, 3, 1).contiguous()
        xs = self.act(xs)
        return self.D_fc2(xs)

    def forward(self, x):
        return self.forward_branch(x)
