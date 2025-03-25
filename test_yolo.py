# import cv2
# import torch

# # 加载 YOLOv5 模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# # 打开视频文件或摄像头
# cap = cv2.VideoCapture(0)  # 0 表示摄像头，或替换为视频文件路径

# def draw_detections(frame, detections, model):
#     for *box, conf, cls in detections:
#         label = f'{model.names[int(cls)]}: {conf:.2f}'
#         cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
#         cv2.putText(frame, label, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 将图像输入模型并获取结果
#     results = model(frame)

#     # 将检测结果转换为 numpy 数组
#     detections = results.xyxy[0].cpu().numpy()

#     # 在图像上绘制检测框
#     draw_detections(frame, detections, model)

#     # 显示结果
#     cv2.imshow('YOLOv5 Detection', frame)

#     # 按 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放资源
# cap.release()
# cv2.destroyAllWindows()





import cv2
import torch
import json
import numpy as np

# 加载 YOLOv5 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# 打开视频文件或摄像头
cap = cv2.VideoCapture(0)  # 0 表示摄像头，或替换为视频文件路径

# 归一化边界框坐标
def normalize_bbox(bbox, image_width, image_height):
    x, y, width, height = bbox
    return [
        x / image_width,
        y / image_height,
        width / image_width,
        height / image_height
    ]

# 绘制检测框
def draw_detections(frame, detections, model):
    for *box, conf, cls in detections:
        label = f'{model.names[int(cls)]}: {conf:.2f}'
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 保存归一化后的检测结果到JSON文件
def save_normalized_detections(detections, image_width, image_height, file_path):
    normalized_detections = []
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        normalized_bbox_coords = normalize_bbox([x1, y1, width, height], image_width, image_height)
        normalized_detections.append({
            "category_id": int(cls),
            "bbox": normalized_bbox_coords,
            "score": float(conf)
        })
    with open(file_path, 'w') as f:
        json.dump(normalized_detections, f, indent=4)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 获取图像的宽度和高度
    image_height, image_width = frame.shape[:2]

    # 将图像输入模型并获取结果
    results = model(frame)

    # 将检测结果转换为 numpy 数组
    detections = results.xyxy[0].cpu().numpy()

    # 在图像上绘制检测框
    draw_detections(frame, detections, model)

    # 保存归一化后的检测结果到JSON文件
    save_normalized_detections(detections, image_width, image_height, 'detections.json')

    # 显示结果
    cv2.imshow('YOLOv5 Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()