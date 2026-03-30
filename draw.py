import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)
        return F.relu(input) * beta - F.relu(-input) * alpha

# 定义SiLU函数
def silu(x):
    return x * torch.sigmoid(x)

# 创建AReLU实例
arelu = AReLU(alpha=0.2, beta=1.5)

# 生成输入数据
x = torch.linspace(-3, 3, 100)
with torch.no_grad():  # 不需要计算梯度
    y_arelu = arelu(x)
    y_relu = F.relu(x)
    y_silu = silu(x)

# 转换为numpy用于绘图
x_np = x.numpy()
y_arelu_np = y_arelu.numpy()
y_relu_np = y_relu.numpy()
y_silu_np = y_silu.numpy()

# 绘制图像
plt.figure(figsize=(10, 7))
plt.plot(x_np, y_arelu_np, label=f'AReLU (α={arelu.alpha.item():.2f}, β={1 + torch.sigmoid(arelu.beta).item():.2f})', linewidth=3)
plt.plot(x_np, y_relu_np, 'r--', label='ReLU', linewidth=2)
plt.plot(x_np, y_silu_np, 'm-', label='SiLU', linewidth=2.5)
plt.plot(x_np, x_np, 'g:', label='Identity', linewidth=1.5)

# 添加图形元素
plt.title('Activation Function Comparison', fontsize=16)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper left')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# 设置坐标轴范围
plt.xlim(-3, 3)
plt.ylim(-1, 3)

# 显示图形
plt.tight_layout()
plt.show()