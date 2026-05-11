import torch
import torch.nn as nn
import math

class CrossAttentionNet(nn.Module):
    """跨模态注意力模型（支持参数配置）"""
    def __init__(self, config=None):
        super().__init__()
        
        # 使用配置参数或默认值
        if config is None:
            config = {}
        
        self.attention_dim = config.get('attention_dim', 8192)
        self.leaky_relu_slope = config.get('leaky_relu_slope', 0.1)
        self.hidden_dims = config.get('hidden_dims', [4096, 2048, 1024, 512, 256, 128, 64, 32])
        
        # 特征提取器
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(9, 9, 9), padding=(3, 3, 3)),
            nn.BatchNorm3d(32, momentum=0.9),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.MaxPool3d(kernel_size=5, stride=5),

            nn.Conv3d(32, 64, kernel_size=(7, 7, 7), padding=(2, 2, 2)),
            nn.BatchNorm3d(64, momentum=0.9),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=(5, 5, 5), padding=(2, 2, 2)),
            nn.BatchNorm3d(128, momentum=0.9),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=(5, 5, 5), padding=(3, 3, 3)),
            nn.BatchNorm3d(256, momentum=0.9),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.MaxPool3d(kernel_size=2, stride=1),

            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(2, 2, 2)),
            nn.BatchNorm3d(512, momentum=0.9),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(2, 2, 2)),
            nn.BatchNorm3d(1024, momentum=0.9),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(1024, 1024, kernel_size=(2, 2, 2), padding=(2, 2, 2)),
            nn.BatchNorm3d(1024, momentum=0.9),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(1024, 2048, kernel_size=(3, 3, 3), padding=(2, 2, 2)),
            nn.BatchNorm3d(2048, momentum=0.9),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(2048, 4096, kernel_size=(3, 3, 3), padding=(2, 2, 2)),
            nn.BatchNorm3d(4096, momentum=0.9),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # 图像特征压缩器
        self.img_compressor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(4096, self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.LeakyReLU(self.leaky_relu_slope)
        )

        # 物理特征编码器
        self.phy_encoder = nn.Sequential(
            nn.Linear(1, self.attention_dim // 4),
            nn.BatchNorm1d(self.attention_dim // 4),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Linear(self.attention_dim // 4, self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.LeakyReLU(self.leaky_relu_slope)
        )

        # 跨模态注意力机制
        self.cross_attention = nn.ModuleDict({
            'query': nn.Linear(self.attention_dim, self.attention_dim),
            'key': nn.Linear(self.attention_dim, self.attention_dim),
            'value': nn.Linear(self.attention_dim, self.attention_dim),
            'out': nn.Linear(self.attention_dim, self.attention_dim)
        })

        self.attention_scale = 1 / torch.sqrt(torch.tensor(self.attention_dim, dtype=torch.float32))

        # 动态构建回归器
        self.regressor = self._build_regressor(self.hidden_dims)
        
        self._initialize_weights()

    def _build_regressor(self, hidden_dims):
        """动态构建回归器层"""
        layers = []
        
        input_dim = self.attention_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(self.leaky_relu_slope))
            input_dim = hidden_dim
        
        # 最后的输出层
        layers.append(nn.Linear(input_dim, 1))
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def visualize_attention(self, data_loader, device, num_samples=5):
        """可视化注意力权重"""
        import matplotlib.pyplot as plt
        
        self.eval()
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6*num_samples))
        
        with torch.no_grad():
            for idx, (data, _, extra) in enumerate(data_loader):
                if idx >= num_samples: 
                    break
                
                # 前向传播获取注意力权重
                img_feat = self.img_compressor(self.features(data.to(device)))
                phy_feat = self.phy_encoder(extra.view(-1, 1).to(device))
                
                Q = self.cross_attention['query'](img_feat)
                K = self.cross_attention['key'](phy_feat)
                attention = torch.sigmoid(torch.sum(Q*K, dim=1)/math.sqrt(self.attention_dim))
                
                # 可视化图像特征
                slice_img = data[0, 0, :, :, 50].cpu().numpy()
                axes[idx, 0].imshow(slice_img, cmap='gray')
                axes[idx, 0].set_title(f"输入图像 (物理特征: {extra[0].item():.2f})")
                
                # 可视化注意力热图
                axes[idx, 1].bar(['注意力权重'], [attention[0].cpu().numpy()], color='tab:orange')
                axes[idx, 1].set_ylim(0, 1)
                axes[idx, 1].set_title(f"注意力值: {attention[0].item():.4f}")
        
        plt.tight_layout()
        return fig

    def forward(self, x, extra_feature=None):
        # 提取图像特征
        x = self.features(x)
        img_features = self.img_compressor(x)

        # 编码物理特征
        if extra_feature is not None:
            phy_features = self.phy_encoder(extra_feature.view(-1, 1))
        else:
            phy_features = torch.zeros_like(img_features)

        # 计算注意力
        Q = self.cross_attention['query'](img_features)
        K = self.cross_attention['key'](phy_features)
        V = self.cross_attention['value'](phy_features)

        attention_scores = torch.sum(Q * K, dim=1, keepdim=True)
        attention_scores = attention_scores * self.attention_scale
        attention_weights = torch.sigmoid(attention_scores)
        
        # 特征融合
        fused_features = attention_weights * V
        fused_features = self.cross_attention['out'](fused_features)
        fused_features += img_features

        # 回归预测
        return self.regressor(fused_features)