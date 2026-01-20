import torch
import torch.nn as nn

class BaseConv3DNet(nn.Module):
    """基础3D CNN模型（支持参数配置）"""
    def __init__(self, config=None):
        super(BaseConv3DNet, self).__init__()
        
        # 使用配置参数或默认值
        if config is None:
            config = {}
        
        self.leaky_relu_negative_slope = config.get('leaky_relu_slope', 0.01)
        self.hidden_dims = config.get('hidden_dims', [1024, 512, 256, 128, 64, 32])
        
        # 特征提取器
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(9, 9, 9), padding=(3, 3, 3)),
            nn.BatchNorm3d(32, momentum=0.9),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
            nn.MaxPool3d(kernel_size=5, stride=5),

            nn.Conv3d(32, 64, kernel_size=(7, 7, 7), padding=(2, 2, 2)),
            nn.BatchNorm3d(64, momentum=0.9),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=(5, 5, 5), padding=(2, 2, 2)),
            nn.BatchNorm3d(128, momentum=0.9),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=(5, 5, 5), padding=(2, 2, 2)),
            nn.BatchNorm3d(256, momentum=0.9),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
            nn.MaxPool3d(kernel_size=2, stride=1),

            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(2, 2, 2)),
            nn.BatchNorm3d(512, momentum=0.9),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(2, 2, 2)),
            nn.BatchNorm3d(1024, momentum=0.9),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(1024, 1024, kernel_size=(2, 2, 2), padding=(2, 2, 2)),
            nn.BatchNorm3d(1024, momentum=0.9),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(1024, 2048, kernel_size=(3, 3, 3), padding=(2, 2, 2)),
            nn.BatchNorm3d(2048, momentum=0.9),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(2048, 4096, kernel_size=(3, 3, 3), padding=(2, 2, 2)),
            nn.BatchNorm3d(4096, momentum=0.9),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # 动态构建回归器
        self.regressor = self._build_regressor(self.hidden_dims)
        
        self._initialize_weights()

    def _build_regressor(self, hidden_dims):
        """动态构建回归器层"""
        layers = []
        
        # 输入维度是固定的
        input_dim = 4096 * 2 * 2 * 2
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope))
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)