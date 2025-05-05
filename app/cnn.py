# cnn.py
# A minimal 2‑class CNN that works with 224×224 RGB images
# (used by both train_cnn.py and main.py)

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Simple 4‑layer conv‑net:
      Conv(3→16) → ReLU → 2×2 MaxPool
      Conv(16→32) → ReLU → 2×2 MaxPool
      Flatten
      FC 32*56*56 → 128 → ReLU
      FC 128 → num_classes
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # 224×224 → 224×224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 224×224 → 112×112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),           # 112×112 → 112×112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 112×112 → 56×56
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                          # 32 × 56 × 56 = 100 352
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # Optional: Kaiming init for conv layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# quick smoke‑test: python cnn.py
if __name__ == "__main__":
    model = CNN()
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print("output shape:", out.shape)          # should be [1, 2]
