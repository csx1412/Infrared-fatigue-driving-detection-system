import os
from PIL import Image


def batch_convert_and_rename(folder_path):
    # 支持的图片格式（不区分大小写）
    supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']

    # 获取所有图片文件（包括原JPG）
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]

    # 按文件名排序（可选：按修改时间排序 os.path.getmtime）
    image_files.sort()

    index = 1  # 起始编号

    for filename in image_files:
        old_path = os.path.join(folder_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        # 生成新文件名（确保不冲突）
        while True:
            new_name = f"{index}.jpg"
            new_path = os.path.join(folder_path, new_name)
            if not os.path.exists(new_path):
                break
            index += 1  # 如果文件名已存在，递增

        try:
            # 如果是JPG/JPEG，直接重命名（不重新编码）
            if file_ext in ('.jpg', '.jpeg'):
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_name}")
            # 其他格式（PNG/GIF等）转换为JPG
            else:
                with Image.open(old_path) as img:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')  # 透明背景转白色
                    img.save(new_path, 'JPEG', quality=95)
                os.remove(old_path)  # 删除原文件
                print(f"转换并重命名: {filename} -> {new_name}")

            index += 1  # 成功处理后递增编号

        except Exception as e:
            print(f"处理失败 {filename}: {e}")


# 使用示例
folder_path = r'E:\bishe\hand-make\make'  # 替换为你的文件夹
batch_convert_and_rename(folder_path)