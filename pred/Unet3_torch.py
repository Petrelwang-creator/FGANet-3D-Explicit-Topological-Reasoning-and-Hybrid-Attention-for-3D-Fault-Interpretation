import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Basic 3D Convolution Block
# ---------------------------------------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------
# CBAM 3D
# ---------------------------------------------------------
class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        # Spatial Attention
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, d, h, w = x.size()

        avg = self.avg_pool(x).view(b, c)
        maxv = self.max_pool(x).view(b, c)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(maxv)).view(b, c, 1, 1, 1)

        x = x * att

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_map, max_map], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial))

        return x * spatial_att


# ---------------------------------------------------------
# Non-local Block 3D
# ---------------------------------------------------------
class NonLocalBlock3D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()

        if inter_channels is None:
            inter_channels = in_channels // 2

        self.g = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.theta = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.out_conv = nn.Conv3d(inter_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.size()

        g_x = self.g(x).view(b, -1, d * h * w)
        theta_x = self.theta(x).view(b, -1, d * h * w)
        phi_x = self.phi(x).view(b, -1, d * h * w)

        att = torch.softmax(torch.bmm(theta_x.transpose(1, 2), phi_x), dim=-1)
        y = torch.bmm(g_x, att.transpose(1, 2))
        y = y.view(b, -1, d, h, w)
        y = self.out_conv(y)

        return x + y


# ---------------------------------------------------------
# UNet3D with CBAM + NonLocal
# ---------------------------------------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=8, use_cbam=True, use_nonlocal=True):
        super().__init__()

        self.use_cbam = use_cbam
        self.use_nonlocal = use_nonlocal

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = ConvBlock3D(base * 4, base * 8)

        # Non-local bottom block
        if self.use_nonlocal:
            self.nonlocal_block = NonLocalBlock3D(base * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)

        self.out_conv = nn.Conv3d(base, out_channels, kernel_size=1)

        # CBAM
        if self.use_cbam:
            self.cbam1 = CBAM3D(base)
            self.cbam2 = CBAM3D(base * 2)
            self.cbam3 = CBAM3D(base * 4)
            self.cbam4 = CBAM3D(base * 8)

    # ---------------------------------------------------------
    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)

        # CBAM
        if self.use_cbam:
            e1 = self.cbam1(e1)
            e2 = self.cbam2(e2)
            e3 = self.cbam3(e3)
            e4 = self.cbam4(e4)

        # Non-local bottom
        if self.use_nonlocal:
            e4 = self.nonlocal_block(e4)

        # Decoder
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out


if __name__ == "__main__":
    # ---------------------------------------------------------
    #快速跑通测试
    # ---------------------------------------------------------
    print("🚀 开始测试 UNet3D 模型...")

    # 1. 实例化模型：默认开启 CBAM (通道和空间注意力) 与 Non-local (全局自注意力) 模块
    # in_channels=1 表示输入单一模态(如单通道地震体)，out_channels=1 表示输出二分类断层概率图
    model = UNet3D(in_channels=1, out_channels=1, base=8, use_cbam=True, use_nonlocal=True)

    # 2. 构造一个随机的 3D 张量模拟输入
    # 形状遵循 PyTorch 3D 卷积标准：(Batch_Size, Channels, Depth, Height, Width)
    dummy_input = torch.randn(2, 1, 64, 64, 64)
    print(f"➡️ 模拟输入张量形状: {dummy_input.shape}")

    # 3. 执行前向传播
    output = model(dummy_input)
    print(f"⬅️ 模型输出张量形状: {output.shape}")

    # 4. 验证输入和输出尺寸是否一致 (UNet 结构应当保持空间维度不变)
    assert dummy_input.shape == output.shape, "❌ 错误：输入和输出的空间形状不一致！"
    print("✅ UNet3D 模型前向传播测试通过！")
