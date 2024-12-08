from torcheval import metrics
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize
import torch
import os


def FID(path_to_real_images:str,
        path_to_fake_images:str,
        device
):

  fid_metric = metrics.FrechetInceptionDistance(device = device)

  transform = Resize((299, 299))
  normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  def preprocess_image(image_path):
      image = Image.open(image_path).convert("RGB")
      tensor = ToTensor()(image)
      tensor = transform(tensor)
      tensor = normalize(tensor)
      tensor = torch.clamp(tensor, 0, 1)
      return tensor

  fake_images = torch.stack([preprocess_image(os.path.join(path_to_fake_images, filename)) for filename in os.listdir(path_to_fake_images)])
  real_images = torch.stack([preprocess_image(os.path.join(path_to_real_images, filename)) for filename in os.listdir(path_to_real_images)])

  fid_metric.update(real_images, is_real=True)
  fid_metric.update(fake_images, is_real=False)

  return fid_metric.compute()