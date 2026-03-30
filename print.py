import pandas as pd
import matplotlib.pyplot as plt
# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取 CSV 文件
data = pd.read_csv('results.csv')
data_2 = pd.read_csv('results1.csv')


# 获取列的数据
mAP_05_data = data['train/cls_loss']
mAP_05_data_2 = data_2['train/cls_loss']


# 绘制曲线
plt.plot(mAP_05_data, label='修改后模型', color='red', linewidth=1)
plt.plot(mAP_05_data_2, label='原模型', color='green', linewidth=1)


# 添加图例
plt.legend(loc='lower right')

# 添加标题和坐标轴标签
plt.xlabel('Epoch')
plt.ylabel('cls_loss(%)')
plt.title('train/cls_loss')

# 网格线
plt.grid(True)

# 保存图像到同目录下
# plt.savefig('mAP_05_curve.png')
plt.show()
