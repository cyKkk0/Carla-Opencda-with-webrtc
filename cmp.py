import json
import numpy as np

# 加载JSON文件
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 计算IoU
def iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[0] + box1[2], box2[0] + box2[2])
    y2_inter = min(box1[1] + box1[3], box2[1] + box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

# 计算mAP50和mAP50-95
def calculate_map(pred_boxes, pred_classes, pred_confidences, ground_truth, iou_thresholds):
    true_positives = np.zeros(len(pred_boxes))
    false_positives = np.zeros(len(pred_boxes))
    false_negatives = len(ground_truth)

    for i, pred_box in enumerate(pred_boxes):
        max_iou = 0
        matched_gt = None

        for gt_box in ground_truth:
            iou_score = iou(pred_box, gt_box)

            if iou_score > max_iou:
                max_iou = iou_score
                matched_gt = gt_box

        if max_iou >= iou_thresholds[0]:
            true_positives[i] = 1
            false_negatives -= 1
        else:
            false_positives[i] = 1

    precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
    recall = true_positives.sum() / (true_positives.sum() + false_negatives)

    return precision, recall

# 加载标准和测试数据
std_data = load_json('std.json')
test_data = load_json('test.json')

# 提取标准数据
std_ground_truth = np.array([[item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3]] for item in std_data])
# std_performance = {
#     "mAP50": 0.85,
#     "mAP50-95": 0.78,
#     "precision": 0.90,
#     "recall": 0.80,
#     "inference_time": 0.02
# }

# 提取测试数据
test_pred_boxes = np.array([[item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3]] for item in test_data])
test_pred_classes = np.array([item['category_id'] for item in test_data])
test_pred_confidences = np.array([item['score'] for item in test_data])

# 计算mAP50和mAP50-95
iou_thresholds = [0.5, 0.95]
precision_50, recall_50 = calculate_map(test_pred_boxes, test_pred_classes, test_pred_confidences, std_ground_truth, [iou_thresholds[0]])
precision_95, recall_95 = calculate_map(test_pred_boxes, test_pred_classes, test_pred_confidences, std_ground_truth, [iou_thresholds[1]])

mAP50 = (precision_50 + recall_50) / 2
mAP50_95 = (precision_95 + recall_95) / 2

# 打印结果
print(f"Test mAP50: {mAP50:.4f}")
print(f"Test mAP50-95: {mAP50_95:.4f}")
print(f"Test Precision: {precision_50:.4f}")
print(f"Test Recall: {recall_50:.4f}")

# 与标准数据比较
# print("\nComparison with Standard Data:")
# print(f"mAP50: {mAP50:.4f} vs {std_performance['mAP50']:.4f}")
# print(f"mAP50-95: {mAP50_95:.4f} vs {std_performance['mAP50-95']:.4f}")
# print(f"Precision: {precision_50:.4f} vs {std_performance['precision']:.4f}")
# print(f"Recall: {recall_50:.4f} vs {std_performance['recall']:.4f}")