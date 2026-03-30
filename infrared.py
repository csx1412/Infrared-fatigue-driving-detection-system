import os
import cv2
import numpy as np
from tqdm import tqdm  # 用于显示进度条


def rgb_to_ir_bw(image, intensity=1.5):
    """黑白红外效果"""
    img_float = image.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    ir = (r * 0.7 + g * 0.2 + b * 0.1) * intensity
    ir = np.clip(ir, 0, 1)
    return (ir * 255).astype(np.uint8)

# def rgb_to_ir(image, intensity=1.5):
#     """将单张RGB图像转换为红外效果"""
#     # 将图像转换为浮点类型以便处理
#     img_float = image.astype(np.float32) / 255.0
#
#     # 分离通道
#     b, g, r = cv2.split(img_float)
#
#     # 红外效果转换 - 增强红色通道，减弱蓝色通道
#     ir = (r * 0.7 + g * 0.2 + b * 0.1) * intensity
#
#     # 限制值在0-1之间
#     ir = np.clip(ir, 0, 1)
#
#     # 转换为8位图像
#     ir_8bit = (ir * 255).astype(np.uint8)
#
#     # 应用伪彩色（可选，使效果更接近真实红外照片）
#     ir_color = cv2.applyColorMap(ir_8bit, cv2.COLORMAP_JET)
#
#     return ir_color


def batch_convert_folder(input_folder, output_folder, intensity=1.5):
    """批量转换文件夹中的所有图片"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有图片文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"在文件夹 {input_folder} 中未找到支持的图片文件")
        return

    print(f"找到 {len(image_files)} 张图片，开始转换...")

    # 处理每张图片
    for filename in tqdm(image_files, desc="处理进度"):
        try:
            # 读取图片
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"无法读取文件: {filename}，跳过")
                continue

            # 转换为红外效果
            ir_img = rgb_to_ir_bw(img, intensity=intensity)

            # 构建输出路径
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}{ext}")

            # 保存图片
            cv2.imwrite(output_path, ir_img)

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

    print(f"转换完成！结果已保存到 {output_folder}")


if __name__ == "__main__":
    # 用户输入设置
    input_folder = input(r'E:\bishe\all\dataset\ces').strip()
    output_folder = input(r'E:\bishe\all\dataset\jieguo').strip()
    intensity = float(input("请输入红外效果强度: ") or "1.5")

    # 执行批量转换
    batch_convert_folder(input_folder, output_folder, intensity)