from PIL import Image
import os

input_folder = r'E:\bishe\hand-make\make'  # 替换为你的图片文件夹路径
output_folder = r'E:\bishe\hand-make\finally'  # 替换为你想保存的路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # 调整尺寸为640x640（会拉伸图像）
            resized_img = img.resize((640, 640))

            # 或者保持比例填充背景（可选）
            # from PIL import ImageOps
            # resized_img = ImageOps.fit(img, (640, 640), method=0, bleed=0.0, centering=(0.5, 0.5))

            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)
            print(f"已处理: {filename}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")