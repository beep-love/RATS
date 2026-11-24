import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

class ResNetEmbedding(nn.Module):
    """
    ResNet backbone + linear projection to an embedding, with L2 normalization.
    Perfectly aligned for Deep Metric Learning baselines.
    """
    def __init__(
        self,
        emb_dim=128,          # Matched to PADS baseline
        pretrained=True,      # Standard ImageNet pretraining
        base_model="resnet50",
        freeze_bn=True        # Freezes BatchNorm stats for DML stability
    ):
        super().__init__()

        # Load Standard ResNet backbone
        if pretrained:
            # Using the modern torchvision weights API
            backbone = getattr(tv, base_model)(
                weights=tv.ResNet50_Weights.IMAGENET1K_V1
            )
        else:
            backbone = getattr(tv, base_model)(weights=None)

        # Replace the original classification head with an Identity layer
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()

        self.backbone = backbone
        
        # Add our custom embedding projection layer
        # For ResNet50, the pooling layer outputs a 2048-dimensional vector
        self.fc = nn.Linear(2048, emb_dim)

        # Apply BatchNorm Freezing
        if freeze_bn:
            self.freeze_batchnorm()
        else:
            print("WARNING: BatchNorm layers are NOT frozen. This may lead to unstable training with smaller Datasets.")

    def freeze_batchnorm(self):
        """
        Locks all BatchNorm2d layers into evaluation mode.
        This forces the network to use the stable ImageNet running means/variances
        instead of calculating highly volatile new stats from small DML batches.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                # Overriding the train method prevents `model.train()` in your 
                # main loop from accidentally unfreezing the batchnorm layers later!
                module.train = lambda _: None

    def forward(self, x):
        h = self.backbone(x)            # [B, 2048]
        z = self.fc(h)                  # [B, emb_dim]
        z = F.normalize(z, p=2, dim=1)  # L2 normalization to project onto hypersphere
        return z