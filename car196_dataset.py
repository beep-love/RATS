import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CAR196Dataset(Dataset):
    def __init__(self, image_paths, labels, image_size=256, crop_size=224, augment=True):
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
        # We store absolute paths directly in this version
        full_path = self.image_paths[idx]
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


def build_cars_dataset(root_dir, image_size=256, crop_size=224):
    """
    Builds the Deep Metric Learning split directly from a merged master folder.
    root_dir should contain 196 subfolders, one for each car model.
    1. Training Pool: First 98 classes (alphabetically sorted)
    2. Testing Pool: Last 98 classes
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Ensure the merged directory exists at {root_dir}")

    # Gather all class folder names and sort alphabetically 
    class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    class_names = sorted([c for c in class_names if not c.startswith('.')])
    
    if len(class_names) != 196:
        print(f"[WARNING] Expected 196 classes, but found {len(class_names)}.")

    # Metric Learning Split (Disjoint Classes)
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for original_idx, class_name in enumerate(class_names):
        # First 98 classes are Train, next 98 are Test
        is_train = original_idx < 98
        
        # Remap labels so both Train and Test are contiguous from 0 to 97
        final_label = original_idx if is_train else original_idx - 98
        
        class_dir = os.path.join(root_dir, class_name)
        
        for img_name in os.listdir(class_dir):
            if not img_name.startswith('.'):  # Ignore hidden files
                full_path = os.path.join(class_dir, img_name)
                
                if is_train:
                    train_paths.append(full_path)
                    train_labels.append(final_label)
                else:
                    test_paths.append(full_path)
                    test_labels.append(final_label)

    # 3. Build Objects
    print(f"Deep Metric Learning CAR196 Split (Merged Source):")
    print(f"  - Train Set: {len(train_paths)} images (Classes: {len(set(train_labels))})")
    print(f"  - Test Set:  {len(test_paths)} images (Classes: {len(set(test_labels))})")

    train_ds = CAR196Dataset(train_paths, train_labels, 
                             image_size=image_size, crop_size=crop_size, augment=True)
    
    test_ds = CAR196Dataset(test_paths, test_labels, 
                            image_size=image_size, crop_size=crop_size, augment=False)

    return train_ds, test_ds


