import os
import cv2
import torch
import json


# 1. 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def normalize_bbox(bbox, image_width, image_height):
    x, y, width, height = bbox
    return [
        x / image_width,
        y / image_height,
        width / image_width,
        height / image_height
    ]

def yol(file_name, image_path):
    # 2. 读取输入图片
    # image_path = os.path.join('/home/bupt/Pictures/', file_name)  # 替换为你的图片路径
    image = cv2.imread(image_path)

    # 3. 预处理图片
    # YOLOv5 默认需要图片为 RGB 格式，而 OpenCV 默认读取为 BGR 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]
    # 4. 进行目标检测
    results = model(image_rgb)

    # 5. 后处理检测结果
    # 获取检测结果的边界框、置信度和类别 ID
    detection_results = []
    for det in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        width = x2 - x1
        height = y2 - y1
        normalized_bbox_coords = normalize_bbox([x1, y1, width, height], image_width, image_height)
        detection_results.append({
            "image_id": 1,  # 假设这是第一张图片
            "category_id": int(cls) + 1,  # YOLOv5 的类别 ID 从 0 开始，COCO 格式从 1 开始
            "bbox": normalized_bbox_coords,
            "score": float(conf)
        })

    # 6. 保存检测结果为 JSON 文件
    output_file = file_name[:-4] + '.json'
    with open(output_file, 'w') as f:
        json.dump(detection_results, f, indent=4)

    print(f"Detection results saved to {output_file}")

    # 将检测结果转换为 numpy 数组
    detections = results.xyxy[0].cpu().numpy()


    def draw_detections(frame, detections, model):
        for *box, conf, cls in detections:
            label = f'{model.names[int(cls)]}: {conf:.2f}'
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # 在图像上绘制检测框
    draw_detections(image, detections, model)
    image_resized = cv2.resize(image, (1600, 1200))
    # 显示结果
    cv2.imshow('YOLOv5 Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # yol('std.jpg', os.path.join('/home/bupt/Pictures/', 'std.jpg'))
    yol('038553.png', os.path.join('./', '038553.png'))