import json
import numpy as np
import os
from tqdm.notebook import tqdm


class Evaluator:

  def __init__(self, dataloader, result_dir):
    self.dl = dataloader
    self.result_dir = result_dir


  def calculate_iou(self, box1, box2):
    # Ensure boxes are 1D arrays with 4 elements
    box1 = box1.reshape(-1)
    box2 = box2.reshape(-1)
    
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return float(intersection / union) if union > 0 else 0.0


  def evaluate(self, prompt):
    # Load predictions from json file
    results_file_path = os.path.join(self.result_dir, f'{prompt}.json')
    with open(results_file_path, 'r') as fp:
      results_data = json.load(fp)

    predictions = results_data['predictions']

    num_matching_labels = 0

    confusion_mat = {}

    # Iterate over zipped samples-predictions (dataloader samples include gt bboxes)
    for sample, prediction in tqdm(zip(self.dl, predictions), desc="Evaluating Predictions", leave=True):

      # Unpack values
      img, bboxes, labels = sample
      h, w = img.shape[2], img.shape[3]
      pred_bboxes = prediction['boxes']
      pred_labels = prediction['phrases']

      # Only bother calculating values if there's matching bboxes
      labels = [label[0] for label in labels] # Unpack wrapper tuple

      for pred_label in pred_labels:
        if pred_label in labels:
          num_matching_labels += 1

      if pred_bboxes == []:
        continue

      # Convert bbox predictions from cx, cy, h, w to x1, y1, x2, y2
      boxes = np.array(pred_bboxes)
      boxes = np.column_stack([
          boxes[:, 0] - boxes[:, 2]/2,  # x1 = cx - w/2
          boxes[:, 1] - boxes[:, 3]/2,  # y1 = cy - h/2
          boxes[:, 0] + boxes[:, 2]/2,  # x2 = cx + w/2
          boxes[:, 1] + boxes[:, 3]/2   # y2 = cy + h/2
      ])

      # Denormalize predicted boxes using image dimensions
      boxes = boxes * np.array([w, h, w, h])

      # For each predicted bounding box, find the closest GT box. If IoU >= 0.5, consider it a false-positive
      for i in range(len(boxes)):
        pred_box = boxes[i]
        pred_label = pred_labels[i]
        for j in range(len(bboxes)):
          gt_box = bboxes[j]
          gt_label = labels[j]

          if self.calculate_iou(pred_box, gt_box) >= 0.5:
            if gt_label not in confusion_mat:
              confusion_mat[gt_label] = {pred_label: 0}
            elif pred_label not in confusion_mat[gt_label]:
              confusion_mat[gt_label][pred_label] = 0
            else:
              confusion_mat[gt_label][pred_label] += 1

    print(f"# Prediction labels matching GT labels: {num_matching_labels}")
    return confusion_mat
