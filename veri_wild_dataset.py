######################################################################################
### Author: Biplav Sharma Regmi <biplav01regmi@gmail.com>
### Date: JAN 2026 - NOV 2026
### Version: 1.0
### Description:
### This code provides functionality to parse the VeRi-wild vehicle re-identification dataset labels
### and create PyTorch Dataset objects for training and validation.
### It includes data augmentation for training and standard preprocessing for validation.
######################################################################################

# ASSUMPTION
#  1. File structure of VeRi-Wild dataset is as follows:
#     veri-wild/
#         images/
#             11542/
#                 119083.jpg
#         train_test_split/
#             train_list_start0.txt
################################################################################


import os
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def parse_veri_wild_list(root_dir, list_file):
    """
    Parses the train_list.txt file from VeRi-Wild.
    Format: image_path vehicle_id camera_id
    Example: 11542/119083.jpg 0 23
    """
    vehicle_data = defaultdict(lambda: {"image_paths": [], "camera_ids": []})
    
    list_path = os.path.join(root_dir, list_file)
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Could not find list file: {list_path}")

    print(f"Parsing VeRi-Wild list file: {list_path}")
    
    with open(list_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        # Extract data
        rel_path = parts[0]
        vehicle_id = int(parts[1])
        camera_id = int(parts[2])
        
        # Construct full path. 
        # usually VeRi-Wild structure is root/images/vehicle_id/img.jpg
        # but the list file usually contains vehicle_id/img.jpg
        full_path = os.path.join(root_dir, "images", rel_path)
        
        vehicle_data[vehicle_id]["image_paths"].append(full_path)
        vehicle_data[vehicle_id]["camera_ids"].append(camera_id)

    return vehicle_data

def split_vehicle_ids(vehicle_data, train_val_split=0.8, seed=33):
    """
    Splits the vehicle IDs into train and validation sets.
    """
    vehicle_ids = sorted(list(vehicle_data.keys()))
    
    # Shuffle IDs to ensure random split
    random.seed(seed)
    random.shuffle(vehicle_ids)

    num_train = int(len(vehicle_ids) * train_val_split)
    train_ids = vehicle_ids[:num_train]
    val_ids   = vehicle_ids[num_train:]

    return train_ids, val_ids

class VeriWildDataset(Dataset):
    def __init__(self, vehicle_data, used_ids, image_size=256, crop_size=224, augment=True):
        self.vehicle_data = vehicle_data
        self.used_ids = used_ids
        
        self.samples = []
        self.labels = []
        self.cameras = []

        # Map original vehicle IDs to a continuous range 0..N-1 (optional, but good practice)
        # However, for PKSampler, we just need the labels to match.
        # We will use the original ID from the text file as the label.
        
        for vid in used_ids:
            info = self.vehicle_data[vid]
            paths = info["image_paths"]
            cams = info["camera_ids"]
            
            for p, c in zip(paths, cams):
                self.samples.append(p)
                self.labels.append(vid)
                self.cameras.append(c)
        
        # Standard ReID Augmentation
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
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # We don't necessarily return camera_id for training, 
        # but it's available in self.cameras[idx] if needed later.

        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or handle error (usually better to crash to know data is bad)
            img = torch.zeros((3, 224, 224)) 

        return img, label

    def get_labels(self):
        return list(self.labels)

def build_veri_wild_datasets(root_dir, list_file, train_val_split=0.8, seed=33, image_size=256, crop_size=224):
    """
    Builder function to be called from train_baseline.py
    """
    # 1. Parse the text file
    vehicle_data = parse_veri_wild_list(root_dir, list_file)
    
    print(f"Total Vehicles found: {len(vehicle_data)}")

    # 2. Split IDs
    train_ids, val_ids = split_vehicle_ids(vehicle_data, train_val_split, seed)

    # 3. Create Datasets
    train_ds = VeriWildDataset(
        vehicle_data, 
        train_ids, 
        image_size=image_size, 
        crop_size=crop_size, 
        augment=True
    )
    
    val_ds = VeriWildDataset(
        vehicle_data, 
        val_ids, 
        image_size=image_size, 
        crop_size=crop_size, 
        augment=False
    )
    
    return train_ds, val_ds

if __name__ == "__main__":
    # Test block
    root = "./data/VeRi-Wild"
    l_file = "train_test_split/train_list_start0.txt"
    
    if os.path.exists(root):
        t_ds, v_ds = build_veri_wild_datasets(root, l_file)
        print(f"Train dataset size: {len(t_ds)}")
        print(f"Val dataset size: {len(v_ds)}")

        print(f"Train ids: {len(set(t_ds.used_ids))}")
        print(f"Val ids:   {len(set(v_ds.used_ids))}")
        
        # Test loading one image
        img, label = t_ds[0]
        print(f"Image shape: {img.shape}, Label: {label}")

        from torch.utils.data import DataLoader

        train_loader = DataLoader(t_ds, batch_size=16, shuffle=True, num_workers=0)

        images, labels = next(iter(train_loader))
        print(f"Batch images shape: {images.shape}")   # [B, 3, H, W]
        print(f"Batch labels shape: {labels.shape}")
        print(f"Unique labels in batch: {labels.unique().tolist()}")