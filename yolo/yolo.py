import onnxruntime as ort
import onnx
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 可配置参数（根据实际模型调整）--------------------------
MODEL_PATH = "test.onnx"  # ONNX 模型路径
IMAGE_PATH = "test.jpg"   # 测试图像路径
CONF_THRESHOLD = 0.5      # 置信度阈值
IOU_THRESHOLD = 0.4       # NMS IOU 阈值
INPUT_SIZE = None         # 若模型输入固定，直接指定（如 (480, 480)），否则自动获取
CLASS_NAMES = {0: "人", 1: "车", 2: "狗"}  # 类别 ID 到名称的映射（根据模型修改）
# 执行设备：CPU 或 GPU（需安装 onnxruntime-gpu 并匹配 CUDA 版本）
EXECUTION_PROVIDERS = ["CPUExecutionProvider"]  # GPU 用 ["CUDAExecutionProvider"]
# --------------------------------------------------------------------------------

def preprocess_image(image_path: str, input_size: Tuple[int, int]) -> np.ndarray:
    """
    图像预处理：读取、格式转换、缩放、归一化、维度调整
    :param image_path: 图像路径
    :param input_size: 模型输入尺寸 (width, height)
    :return: 预处理后的输入张量 (1, C, H, W)
    """
    # 读取图像并检查是否成功
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}（检查路径是否正确、文件是否损坏）")
    
    # BGR -> RGB（YOLO 模型默认输入为 RGB）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 保持长宽比缩放（避免图像拉伸），用黑色填充空白部分
    h, w = image.shape[:2]
    target_w, target_h = input_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image, (new_w, new_h))
    
    # 创建空白画布，填充图像
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    offset_x, offset_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w, :] = image_resized
    
    # 归一化 + 维度转换 (H, W, C) -> (1, C, H, W)
    canvas = canvas.astype(np.float32) / 255.0
    canvas = np.transpose(canvas, (2, 0, 1))  # (C, H, W)
    canvas = np.expand_dims(canvas, axis=0)   # (1, C, H, W)
    return canvas, (scale, offset_x, offset_y)  # 返回缩放比例和偏移量（用于后处理修正坐标）

def postprocess_output(
    output: List[np.ndarray], 
    input_size: Tuple[int, int],
    image_shape: Tuple[int, int],
    scale: float,
    offset_x: int,
    offset_y: int,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.4
) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """
    后处理：解析检测结果、过滤低置信度、NMS、修正坐标
    :param output: 模型推理输出
    :param input_size: 模型输入尺寸 (width, height)
    :param image_shape: 原始图像尺寸 (height, width)
    :param scale: 预处理时的缩放比例
    :param offset_x: 预处理时的水平偏移量
    :param offset_y: 预处理时的垂直偏移量
    :return: 过滤后的检测框、置信度、类别 ID
    """
    # 适配不同 YOLO 模型输出（YOLOv5: [1, 25200, 85], YOLOv8: [1, 84, 640, 640]）
    outputs = output[0]
    if len(outputs.shape) == 4:  # YOLOv8 格式 (N, 84, H, W) -> 转为 (N, H*W, 84)
        outputs = outputs.transpose(0, 2, 3, 1).reshape(1, -1, 84)
    
    # 解析输出（假设最后一维为：x, y, w, h, conf, cls_0, cls_1, ...）
    num_classes = outputs.shape[-1] - 5
    boxes, scores, class_ids = [], [], []
    
    for detection in outputs[0]:
        x, y, w, h, conf = detection[:5]
        cls_probs = detection[5:5+num_classes]
        
        # 过滤低置信度
        if conf < conf_threshold:
            continue
        
        # 获取类别
        class_id = np.argmax(cls_probs)
        score = conf * cls_probs[class_id]
        if score < conf_threshold:
            continue
        
        # 修正坐标（从模型输入尺寸映射回原始图像尺寸）
        # 1. 从模型输出的归一化坐标转为输入尺寸的像素坐标
        x = x * input_size[0] - offset_x
        y = y * input_size[1] - offset_y
        w = w * input_size[0]
        h = h * input_size[1]
        
        # 2. 修正缩放和偏移（还原到原始图像尺寸）
        x = x / scale
        y = y / scale
        w = w / scale
        h = h / scale
        
        # 3. 转换为左上角 (x1, y1) 和右下角 (x2, y2)
        x1 = max(0, x - w / 2)
        y1 = max(0, y - h / 2)
        x2 = min(image_shape[1], x + w / 2)
        y2 = min(image_shape[0], y + h / 2)
        
        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        class_ids.append(int(class_id))
    
    # 非极大值抑制（NMS）
    if boxes:
        boxes_arr = np.array(boxes)
        indices = cv2.dnn.NMSBoxes(
            boxes_arr[:, :4], scores, conf_threshold, iou_threshold,
            score_threshold=conf_threshold, nms_threshold=iou_threshold
        )
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = [boxes[i] for i in indices]
            scores = [scores[i] for i in indices]
            class_ids = [class_ids[i] for i in indices]
    
    return boxes, scores, class_ids

def visualize_results(
    image_path: str,
    boxes: List[np.ndarray],
    scores: List[float],
    class_ids: List[int],
    class_names: Dict[int, str],
    save_path: Optional[str] = "result.jpg"
) -> None:
    """
    可视化检测结果：绘制边界框、类别标签、置信度
    :param image_path: 原始图像路径
    :param save_path: 结果保存路径（None 不保存）
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    
    # 绘制每个检测框
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        # 绘制矩形框（绿色，线宽 2）
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 类别名称（若未定义，显示 ID）
        class_name = class_names.get(class_id, f"Class {class_id}")
        # 标签文本
        label = f"{class_name}: {score:.2f}"
        # 绘制标签背景（半透明黑色）
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - label_size[1] - 5)
        cv2.rectangle(
            image, (x1, label_y1), (x1 + label_size[0], y1 - 1),
            (0, 0, 0), -1  # 填充黑色背景
        )
        # 绘制标签文本（白色）
        cv2.putText(
            image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    # 显示结果
    cv2.imshow("Detection Results", image)
    print(f"检测结果已显示，按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    if save_path:
        cv2.imwrite(save_path, image)
        print(f"检测结果已保存到：{save_path}")

def load_onnx_model(model_path: str, providers: List[str]) -> Tuple[ort.InferenceSession, str, str, Tuple[int, int]]:
    """
    加载 ONNX 模型并获取输入/输出信息
    :param model_path: 模型路径
    :param providers: 执行设备列表
    :return: 推理会话、输入名称、输出名称、输入尺寸 (width, height)
    """
    # 验证模型完整性
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print("模型验证通过，结构合法")
    except Exception as e:
        raise RuntimeError(f"模型验证失败：{str(e)}")
    
    # 创建推理会话
    session = ort.InferenceSession(model_path, providers=providers)
    
    # 获取输入/输出信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    input_name = input_info.name
    output_name = output_info.name
    input_shape = input_info.shape  # 格式通常为 (1, 3, H, W) 或 (3, H, W)
    
    # 解析输入尺寸（处理动态维度）
    if len(input_shape) == 4:
        h, w = input_shape[2], input_shape[3]
    elif len(input_shape) == 3:
        h, w = input_shape[1], input_shape[2]
    else:
        raise ValueError(f"不支持的输入形状：{input_shape}（仅支持 3 或 4 维）")
    
    # 处理动态维度（如 -1 或字符串）
    if isinstance(h, (str, int)) and (h == -1 or isinstance(h, str)):
        if INPUT_SIZE is None:
            raise ValueError("模型输入为动态维度，请在参数中指定 INPUT_SIZE（如 (480, 480)）")
        w, h = INPUT_SIZE
    input_size = (int(w), int(h))
    
    print(f"模型输入信息：名称={input_name}, 形状={input_shape}, 实际输入尺寸={input_size}")
    print(f"模型输出信息：名称={output_name}, 形状={output_info.shape}")
    return session, input_name, output_name, input_size

def main():
    try:
        # 1. 加载模型
        session, input_name, output_name, input_size = load_onnx_model(MODEL_PATH, EXECUTION_PROVIDERS)
        
        # 2. 读取原始图像并获取尺寸
        raw_image = cv2.imread(IMAGE_PATH)
        if raw_image is None:
            raise FileNotFoundError(f"无法读取图像：{IMAGE_PATH}")
        image_shape = raw_image.shape[:2]  # (height, width)
        
        # 3. 图像预处理
        preprocessed_img, (scale, offset_x, offset_y) = preprocess_image(IMAGE_PATH, input_size)
        print(f"预处理完成，输入张量形状：{preprocessed_img.shape}")
        
        # 4. 模型推理
        print("开始推理...")
        output = session.run([output_name], {input_name: preprocessed_img})
        print(f"推理完成，输出形状：{output[0].shape}")
        
        # 5. 后处理结果
        boxes, scores, class_ids = postprocess_output(
            output, input_size, image_shape, scale, offset_x, offset_y,
            conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD
        )
        
        # 6. 输出检测结果
        print(f"\n检测结果汇总：")
        print(f"共检测到 {len(boxes)} 个对象")
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box
            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
            print(f"对象 {i+1}：类别={class_name}, 置信度={score:.4f}, 位置=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
        
        # 7. 可视化结果
        if boxes:
            visualize_results(IMAGE_PATH, boxes, scores, class_ids, CLASS_NAMES)
        else:
            print("未检测到任何对象（可降低置信度阈值重试）")
    
    except Exception as e:
        print(f"程序执行失败：{str(e)}")
    finally:
        cv2.destroyAllWindows()  # 确保窗口关闭

if __name__ == "__main__":
    main()