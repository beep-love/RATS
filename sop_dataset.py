import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict

class SOPDataset(Dataset):
    def __init__(self, root_dir, samples, labels, image_size=256, crop_size=224, augment=True):
        """
        Generic SOP Dataset.
        Args:
            root_dir: Root of dataset.
            samples: List of relative image paths.
            labels: List of integer class labels.
        """
        self.root_dir = root_dir
        self.samples = samples
        self.labels = labels
        
        # Standard Transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop((crop_size, crop_size)),
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
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]
        full_path = os.path.join(self.root_dir, rel_path)
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

def build_sop_dataset(root_dir, image_size=256, crop_size=224):
    """
    1. Loads Ebay_train.txt -> 100% of images go to Train Dataset.
    2. Loads Ebay_test.txt -> Held-out Test Set (used for Val & Test).
    """
    
    # -------------------------------------------------------
    # Parse Official Training List
    # -------------------------------------------------------
    train_list_path = os.path.join(root_dir, "Ebay_train.txt")
    if not os.path.exists(train_list_path):
        raise FileNotFoundError(f"Missing {train_list_path}")
    
    # SOP files have a header line. Read with pandas.
    df_train = pd.read_csv(train_list_path, sep=r'\s+', engine='python')

    df_test = pd.read_csv(os.path.join(root_dir, "Ebay_test.txt"), sep=r'\s+', engine='python')

    train_paths = df_train['path'].tolist()
    train_labels_raw = df_train['class_id'].tolist()

    test_paths = df_test['path'].tolist()
    test_labels_raw = df_test['class_id'].tolist()

    # -------------------------------------------------------
    # Independent Label Remapping (0 .. N-1)
    # -------------------------------------------------------
    # We must remap train and test INDEPENDENTLY so both start at 0.

    def remap_labels(labels_list):
        unique_labels = sorted(list(set(labels_list)))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return [label_map[l] for l in labels_list]

    train_labels = remap_labels(train_labels_raw)
    test_labels = remap_labels(test_labels_raw)
    
    # -------------------------------------------------------
    # Create Datasets
    # -------------------------------------------------------
    print(f"SOP Split Summary:")
    print(f"  - Train: {len(train_paths)} images (Classes: {len(set(train_labels))})")
    print(f"  - Test:  {len(test_paths)} images (Classes: {len(set(test_labels))})")

    train_ds = SOPDataset(root_dir, train_paths, train_labels, 
                          image_size=image_size, crop_size=crop_size, augment=True)
    
    test_ds = SOPDataset(root_dir, test_paths, test_labels, 
                         image_size=image_size, crop_size=crop_size, augment=False)
    
    return train_ds, test_ds