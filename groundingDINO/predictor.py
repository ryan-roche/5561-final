import json
import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
from tqdm import tqdm

class Predictor:

  def __init__(self, config_path, weights_path, box_threshold=0.35, text_threshold=0.25):
    self.model = load_model(config_path, weights_path)
    self.box_threshold = box_threshold
    self.text_threshold = text_threshold

  
  def predict(self, image, prompt):
    boxes, logits, phrases = predict(
      model=self.model,
      image=image,
      caption=prompt,
      box_threshold=self.box_threshold,
      text_threshold=self.text_threshold
    )

    return boxes, logits, phrases


def batch_predict(predictor, dataloader, prompts, prompt_names, results_dir):
  
  def _predict_for_prompt(prompt, prompt_name):

    predictions = []
    for sample in tqdm(dataloader, desc=f"Running Inference for {prompt_name}"):
      image, gt_bboxes = sample

      boxes, logits, phrases = predictor.predict(
        image=image.squeeze(0),
        prompt=prompt
      )

      predictions.append({'boxes': boxes.tolist(), 'logits': logits.tolist(), 'phrases': phrases})

    results_path = os.path.join(results_dir, f"{prompt_name}.json")

    with open(results_path, 'w') as f:
      json_dict = {
        'model_params': {'box_threshold': predictor.box_threshold, 'text_threshold': predictor.text_threshold},
        'prompt': prompt,
        'predictions': predictions
      }

      json.dump(json_dict, f, indent=4)  # Use indent for pretty formatting

  for idx, prompt in enumerate(prompts):
    
    prompt_name = prompt_names[idx]
    _predict_for_prompt(prompt, prompt_name)

