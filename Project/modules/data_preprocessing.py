import json
import os
import scipy.io
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def data_reallocation(
    data_path: str,
    images_path: str ,
    output_path:str
):

    labels = scipy.io.loadmat(os.path.join(data_path, 'imagelabels.mat'))['labels'][0]
    with open(data_path + '/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    labels = [cat_to_name[str(label)] for label in labels]

    setid = scipy.io.loadmat(os.path.join(data_path, 'setid.mat'))

    train_ids = setid['trnid'][0]
    val_ids = setid['valid'][0]
    test_ids = setid['tstid'][0]

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_path, split)
        os.makedirs(split_path, exist_ok=True)
        for label in labels:  
            os.makedirs(os.path.join(split_path, str(label)), exist_ok=True)

    def move_images(ids, split):
        for img_id in ids:
            label = labels[img_id - 1]
            src = os.path.join(images_path, f'image_{img_id:05d}.jpg')
            dst = os.path.join(output_path, split, str(label), f'image_{img_id:05d}.jpg')
            shutil.copy(src, dst)

    move_images(train_ids, 'train')
    move_images(val_ids, 'val')
    move_images(test_ids, 'test')

    print(f"Images reallocated successfully to folder {output_path}")

def create_datasets(output_path: str, batch_size: int):
    train_dir = output_path + '/train'
    val_dir = output_path + '/val'
    test_dir = output_path + '/test'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
      	transforms.RandomHorizontalFlip(p=0.5),
    	transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    	transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    return train_dataset, val_dataset, test_dataset