# 导入必要的库
import cv2
import gradio as gr
from ultralytics import YOLO
import numpy as np
import tempfile
import time
from PIL import Image, ImageDraw, ImageFont  # 用于处理中文文本

# 加载预训练的YOLO模型
model = YOLO("last8.pt")

# 全局变量用于控制摄像头
camera_active = False
cap = None

# 疲劳检测相关变量
closed_eye_start_time = None
fatigue_detected = False
FATIGUE_THRESHOLD = 2  # 2秒

# 加载中文字体（确保系统中有这个字体文件）
try:
    font = ImageFont.truetype("simhei.ttf", 40)  # 使用黑体
except:
    font = ImageFont.load_default()
    print("警告: 未找到中文字体，将使用默认字体")

# 自定义CSS样式
custom_css = """
/* 主容器样式 */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* 标题样式 */
.title {
    text-align: center;
    color: #2c3e50;
    font-weight: 700;
    margin-bottom: 20px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

/* 标签页样式 */
.tab-nav {
    background: rgba(255,255,255,0.8) !important;
    border-radius: 10px !important;
    padding: 5px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}

/* 按钮样式 */
button {
    background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 8px rgba(0,0,0,0.15) !important;
}

/* 输入输出区域样式 */
.input-image, .output-image, .input-video, .output-video {
    border-radius: 15px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    border: 2px solid rgba(255,255,255,0.5) !important;
}

/* 标签样式 */
label {
    font-weight: 600 !important;
    color: #2c3e50 !important;
    margin-bottom: 8px !important;
}

/* 标签页内容样式 */
.tab-item {
    background: rgba(255,255,255,0.7) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
}

/* 按钮容器样式 */
.button-container {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-top: 20px;
}
"""


def add_fatigue_warning(frame):
    """在帧上添加疲劳警告（支持中文）"""
    # 将OpenCV图像转换为PIL图像
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    warning_text = "疲劳警告! 请休息!"

    # 使用textbbox获取文本尺寸（新方法）
    left, top, right, bottom = draw.textbbox((0, 0), warning_text, font=font)
    text_width = right - left
    text_height = bottom - top

    # 计算文本位置（居中）
    x = (frame.shape[1] - text_width) // 2
    y = (frame.shape[0] - text_height) // 2

    # 添加背景框
    draw.rectangle(
        [(x - 10, y - 10), (x + text_width + 10, y + text_height + 10)],
        fill=(0, 0, 0)  # 黑色背景
    )

    # 添加文本
    draw.text((x, y), warning_text, font=font, fill=(255, 0, 0))  # 红色文字

    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)


def detect_image(input_image):
    """处理图片检测的函数"""
    image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    results = model(image)
    annotated_image = results[0].plot()

    # 检查是否有closed_eye标签
    detected_labels = [model.names[int(box.cls)] for result in results for box in result.boxes]
    if "Closed Eye" in detected_labels:
        annotated_image = add_fatigue_warning(annotated_image)

    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


def detect_video(input_video):
    """处理视频检测的函数"""
    global closed_eye_start_time, fatigue_detected

    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = temp_output.name

    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # 检查当前帧是否有closed_eye
        current_frame_has_closed_eye = any(model.names[int(box.cls)] == "Closed Eye"
                                           for result in results for box in result.boxes)

        # 更新疲劳检测状态
        if current_frame_has_closed_eye:
            if closed_eye_start_time is None:
                closed_eye_start_time = time.time()
            elif time.time() - closed_eye_start_time >= FATIGUE_THRESHOLD:
                fatigue_detected = True
        else:
            closed_eye_start_time = None
            fatigue_detected = False

        # 如果检测到疲劳，添加警告
        if fatigue_detected:
            annotated_frame = add_fatigue_warning(annotated_frame)

        out.write(annotated_frame)

    cap.release()
    out.release()
    closed_eye_start_time = None
    fatigue_detected = False
    return output_path


def start_camera():
    """启动摄像头并返回生成器函数"""
    global camera_active, cap, closed_eye_start_time, fatigue_detected
    camera_active = True
    cap = cv2.VideoCapture(0)
    closed_eye_start_time = None
    fatigue_detected = False

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # 检查当前帧是否有closed_eye
        current_frame_has_closed_eye = any(model.names[int(box.cls)] == "Closed Eye"
                                           for result in results for box in result.boxes)

        # 更新疲劳检测状态
        if current_frame_has_closed_eye:
            if closed_eye_start_time is None:
                closed_eye_start_time = time.time()
            elif time.time() - closed_eye_start_time >= FATIGUE_THRESHOLD:
                fatigue_detected = True
        else:
            closed_eye_start_time = None
            fatigue_detected = False

        # 如果检测到疲劳，添加警告
        if fatigue_detected:
            annotated_frame = add_fatigue_warning(annotated_frame)

        yield cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    cap.release()


def stop_camera():
    """停止摄像头"""
    global camera_active, closed_eye_start_time, fatigue_detected
    camera_active = False
    closed_eye_start_time = None
    fatigue_detected = False
    return None


# 创建Gradio界面
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <div class="title">
        <h1>🚀 YOLO 目标检测系统</h1>
        <p>基于YOLOv11的实时目标检测平台</p>
    </div>
    """)

    with gr.Tabs():
        # 图片检测标签页
        with gr.TabItem("📷 图片检测"):
            gr.Markdown("### 上传图片进行目标检测")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="输入图片", elem_classes="input-image")
                    with gr.Row():
                        image_button = gr.Button("开始检测", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="检测结果", elem_classes="output-image")

        # 视频检测标签页
        with gr.TabItem("🎥 视频检测"):
            gr.Markdown("### 上传视频进行目标检测")
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="输入视频", elem_classes="input-video")
                    with gr.Row():
                        video_button = gr.Button("开始检测", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="检测结果", elem_classes="output-video")

        # 实时摄像头检测标签页
        with gr.TabItem("🌐 实时检测"):
            gr.Markdown("### 实时摄像头目标检测")
            webcam_output = gr.Image(label="实时检测画面", streaming=True, elem_classes="output-image")
            with gr.Row(elem_classes="button-container"):
                start_button = gr.Button("▶️ 启动摄像头", variant="primary")
                stop_button = gr.Button("⏹️ 停止摄像头", variant="secondary")

    # 绑定事件处理函数
    image_button.click(detect_image, inputs=image_input, outputs=image_output)
    video_button.click(detect_video, inputs=video_input, outputs=video_output)
    start_button.click(start_camera, outputs=webcam_output)
    stop_button.click(stop_camera, outputs=webcam_output)

if __name__ == "__main__":
    # 确保正确释放摄像头资源
    try:
        demo.launch(
            server_name="127.0.0.1",  # 只允许本机访问
            server_port=8080,  # 默认端口
            share=False,  # 不创建公开链接
            favicon_path=None
        )
    finally:
        if cap is not None:
            cap.release()