import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, image_paths, labels, image_size=256, crop_size=224, augment=True):
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.labels = labels
        
        # Transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        full_path = os.path.join(self.root_dir, 'images', rel_path)
        label = self.labels[idx]

        try:
            img = Image.open(full_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            img = torch.zeros((3, 224, 224))
            
        return img, label
    
    def get_labels(self):
        return self.labels

def build_cub_dataset(root_dir, image_size=256, crop_size=224):
    """
    Splits CUB data for Deep Metric Learning (Zero-Shot / Disjoint Split):
    1. Training Pool: Classes 1 to 100 (100% of images used for training)
    2. Testing Pool: Classes 101 to 200 (Completely unseen during training)
    """
    
    # 1. Parse Metadata
    images_txt = os.path.join(root_dir, 'images.txt')
    labels_txt = os.path.join(root_dir, 'image_class_labels.txt')
    
    def read_kv(path, val_type=str):
        d = {}
        with open(path, 'r') as f:
            for line in f:
                k, v = line.strip().split()
                d[k] = val_type(v)
        return d

    id_to_path  = read_kv(images_txt, str)
    id_to_label = read_kv(labels_txt, int)

    # 2. Metric Learning Split (Disjoint Classes)
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []

    for img_id, label in id_to_label.items():
        # CUB labels in the text file are 1-indexed (1 to 200)
        if label <= 100:
            train_paths.append(id_to_path[img_id])
            train_labels.append(label - 1)  # 0-indexed (0 to 99)
        else:
            test_paths.append(id_to_path[img_id])
            test_labels.append(label - 1)   # 0-indexed (100 to 199)

    # 3. Build Objects
    print(f"Deep Metric Learning CUB Split:")
    print(f"  - Train Set: {len(train_paths)} images (Classes: {len(set(train_labels))})")
    print(f"  - Test Set:  {len(test_paths)} images (Classes: {len(set(test_labels))})")

    train_ds = CUB200Dataset(root_dir, train_paths, train_labels, 
                             image_size=image_size, crop_size=crop_size, augment=True)
    
    test_ds = CUB200Dataset(root_dir, test_paths, test_labels, 
                            image_size=image_size, crop_size=crop_size, augment=False)

    return train_ds, test_ds

if __name__ == "__main__":
    # Example usage
    root_directory = "./data/CUB_200/CUB_200_2011/"
    train_dataset, val_dataset = build_cub_dataset(root_directory)