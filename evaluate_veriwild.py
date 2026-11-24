import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import ResNetEmbedding

# Default data root; a RAM-disk copy can be supplied via VERI_WILD_ROOT.
DEFAULT_ROOT = os.environ.get("VERI_WILD_ROOT", "./data/VeRi-Wild")

# Official test splits: number of identities -> split name used in the tables.
TEST_SPLITS = {3000: "small", 5000: "medium", 10000: "large"}


# ---------------------------------------------------------
# VeRi-Wild Data Handling
# ---------------------------------------------------------
def parse_file(list_file):
    """Parses the VeRi-Wild txt files."""
    data = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, vid_str, cam_str = line.split()
            data.append((img_path, int(vid_str), int(cam_str)))
    return data


class VeRiWildEvalDataset(Dataset):
    """Returns (image, vehicle_id, camera_id)."""
    def __init__(self, img_root, data_list, transform=None):
        self.img_root = img_root
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        rel_path, vid, camid = self.data_list[idx]
        full_path = os.path.join(self.img_root, "images", rel_path)
        # No fallback image: if a file cannot be read, evaluation must stop.
        # Substituting a blank image would make all embeddings identical and
        # produce meaningless metrics that look like a normal result.
        img = Image.open(full_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, vid, camid


def extract_features(model, dataloader, device):
    """Extracts features, vehicle IDs and camera IDs for a loader."""
    model.eval()
    features, pids, camids = [], [], []
    with torch.no_grad():
        for imgs, vids, cids in dataloader:
            feats = model(imgs.to(device))        # already L2-normalized by the model
            features.append(feats.cpu())
            pids.extend(vids.numpy())
            camids.extend(cids.numpy())
    return torch.cat(features, dim=0), torch.tensor(pids), torch.tensor(camids)


# ---------------------------------------------------------
# Cross-camera ReID evaluation
# ---------------------------------------------------------
def evaluate_reid(q_feats, q_pids, q_camids, g_feats, g_pids, g_camids, max_rank=50):
    """CMC and mAP, excluding gallery images with the same ID AND same camera."""
    distmat = torch.cdist(q_feats, g_feats, p=2).numpy()

    q_pids, q_camids = q_pids.numpy(), q_camids.numpy()
    g_pids, g_camids = g_pids.numpy(), g_camids.numpy()

    num_q = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP = [], []
    num_valid_q = 0.

    for q_idx in range(num_q):
        order = indices[q_idx]
        remove = (g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx])
        raw_cmc = matches[q_idx][~remove]

        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum() / (np.arange(len(raw_cmc)) + 1.0)
        all_AP.append((tmp_cmc * raw_cmc).sum() / num_rel)

    assert num_valid_q > 0, "Error: No valid queries found!"

    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    return all_cmc, float(np.mean(all_AP))


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def run_veriwild_evaluation(checkpoint_path, data_root=DEFAULT_ROOT,
                            batch_size=256, num_workers=8, verbose=True):
    """
    Evaluate one checkpoint on the three VeRi-Wild test splits.

    Returns a dict of dicts, metrics in percent:
        {"small":  {"R@1": .., "R@5": .., "R@10": .., "mAP": ..},
         "medium": {...},
         "large":  {...}}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = ResNetEmbedding(emb_dim=128, pretrained=False,
                            base_model="resnet50", freeze_bn=False).to(device)
    if verbose:
        print(f"Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    results = {}
    for size, split in TEST_SPLITS.items():
        if verbose:
            print(f"\n=== VeRi-Wild {split} split ({size} IDs) ===")

        split_dir = os.path.join(data_root, "train_test_split")
        query_data = parse_file(os.path.join(split_dir, f"test_{size}_id_query.txt"))
        gallery_data = parse_file(os.path.join(split_dir, f"test_{size}_id.txt"))

        q_loader = DataLoader(VeRiWildEvalDataset(data_root, query_data, test_transform),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
        g_loader = DataLoader(VeRiWildEvalDataset(data_root, gallery_data, test_transform),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

        qf, q_pids, q_camids = extract_features(model, q_loader, device)
        gf, g_pids, g_camids = extract_features(model, g_loader, device)

        cmc, mAP = evaluate_reid(qf, q_pids, q_camids, gf, g_pids, g_camids)

        results[split] = {
            "R@1": float(cmc[0]) * 100,
            "R@5": float(cmc[4]) * 100,
            "R@10": float(cmc[9]) * 100,
            "mAP": mAP * 100,
        }

        if verbose:
            for k, v in results[split].items():
                print(f" {k:<5}: {v:.2f}%")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=DEFAULT_ROOT)
    args = parser.parse_args()

    run_veriwild_evaluation(args.checkpoint, args.data_root)