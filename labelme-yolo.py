import json
import os
import cv2


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def convert_labelme_to_yolo(labelme_json, image_path, class_list, output_txt):
    """
    将LabelMe的JSON标注转换为YOLO格式的TXT标注文件。
    :param labelme_json: LabelMe生成的JSON文件路径
    :param image_path: 图像文件路径，用于获取图像尺寸进行归一化
    :param class_list: 类别列表，用于映射类别ID
    :param output_txt: 输出的YOLO格式TXT文件路径
    """
    print(f"Converting {labelme_json} to YOLO format...")
    with open(labelme_json, 'r') as f:
        labelme_data = json.load(f)

    img = cv2.imread(image_path)
    height, width, _ = img.shape
    # height, width = labelme_data['imageHeight'], labelme_data['imageWidth']
    # 初始化YOLO格式的标注列表
    yolo_annotations = []

    for shape in labelme_data['shapes']:
        # 获取类别名称并映射到ID
        class_name = shape['label']
        if class_name in class_list:
            class_id = class_list.index(class_name)
            # class_id = int(class_name)
        else:
            raise ValueError(f"Class '{class_name}' not found in class list.")

        # 计算bounding box的坐标(x_min, y_min, x_max, y_max)
        x_min, y_min = shape['points'][0]
        x_max, y_max = shape['points'][1]

        # 转换为YOLO格式的归一化坐标(x_center, y_center, width, height)
        x_center = (x_min + x_max) / 2.0 / width
        y_center = (y_min + y_max) / 2.0 / height
        bbox_width = (x_max - x_min) / width
        bbox_height = (y_max - y_min) / height

        # 添加到标注列表
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

    # 写入TXT文件
    with open(output_txt, 'w') as f:
        f.write('\n'.join(yolo_annotations))


# 示例使用
labelme_json_path = r'E:\bishe\111\annotations'  # json标签
image_path = r'E:\bishe\111\images'  # 图像数据
class_list = ['closed_eye', 'closed_mouth', 'opened_eye', 'opened_mouth']  # 根据实际情况定义类别
output_txt_path = r'E:\bishe\111\labels'
json_path_list = []
img_path_list = []
json_path_list = getFileList(labelme_json_path, json_path_list, ext='.json')
img_path_list = getFileList(image_path, img_path_list, ext='.jpg')  # 如果是png图像则写入.png
json_path_list = sorted(json_path_list)
img_path_list = sorted(img_path_list)
for i in range(len(json_path_list)):
    output_txt = os.path.join(output_txt_path, os.path.basename(json_path_list[i]).split('.j')[0] + '.txt')
    convert_labelme_to_yolo(json_path_list[i], img_path_list[i], class_list, output_txt)