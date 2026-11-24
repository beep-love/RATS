import argparse
import torch
from torch.utils.data import DataLoader

# --- Import your modules ---
from cub_200_dataset import build_cub_dataset
from sop_dataset import build_sop_dataset
from car196_dataset import build_cars_dataset
from model import ResNetEmbedding
from metrics import compute_embeddings, compute_retrieval_metrics, nmi_gpu

# --- Fixed settings (must match training) ---
IMAGE_SIZE = 256
CROP_SIZE = 224
EMB_DIM = 128
MODEL_TYPE = "resnet50"

# Per-dataset protocol: data root, Recall@K depths, dataset builder.
DATASET_SPECS = {
    "CUB200": {
        "root": "./data/CUB_200/CUB_200_2011",
        "ranks": [1, 2, 4],
        "builder": build_cub_dataset,
    },
    "CARS196": {
        "root": "./data/CAR196",
        "ranks": [1, 2, 4],
        "builder": build_cars_dataset,
    },
    "SOP": {
        "root": "./data/Stanford_Online_Products",
        "ranks": [1, 10, 100],
        "builder": build_sop_dataset,
    },
}


def load_model(checkpoint_path, device, freeze_bn=True):
    """Build the embedding network and load a trained checkpoint."""
    model = ResNetEmbedding(
        emb_dim=EMB_DIM,
        pretrained=False,
        base_model=MODEL_TYPE,
        freeze_bn=freeze_bn,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_best_model(dataset, checkpoint_path, data_root=None,
                        batch_size=256, num_workers=8, verbose=True):
    """
    Evaluate one checkpoint on the test split of CUB200, CARS196 or SOP.

    Returns a dict of metrics in percent, e.g. for CUB200:
        {"R@1": .., "R@2": .., "R@4": .., "NMI": .., "mAP": ..}
    """
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset}. Expected one of {list(DATASET_SPECS)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = DATASET_SPECS[dataset]
    root = data_root or spec["root"]
    ranks = spec["ranks"]

    if verbose:
        print(f"\nLoading {dataset} from: {root}")
    _, test_ds = spec["builder"](root, image_size=IMAGE_SIZE, crop_size=CROP_SIZE)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    if verbose:
        print(f"Test set ready: {len(test_ds)} images from unseen classes.")
        print(f"Loading weights from: {checkpoint_path}")

    model = load_model(checkpoint_path, device, freeze_bn=True)

    emb, labels = compute_embeddings(model, test_loader, device)
    if emb is None:
        raise RuntimeError("No embeddings extracted; check the test loader.")

    r_a, r_b, r_c, mAP = compute_retrieval_metrics(emb, labels, ranks=ranks)
    nmi = nmi_gpu(emb, labels)          # k-means seed fixed inside metrics.py

    results = {
        f"R@{ranks[0]}": r_a * 100,
        f"R@{ranks[1]}": r_b * 100,
        f"R@{ranks[2]}": r_c * 100,
        "NMI": nmi * 100,
        "mAP": mAP * 100,
    }

    del model, emb, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print("\n" + "=" * 40)
        print(f" FINAL {dataset} TEST SET BENCHMARKS")
        print("=" * 40)
        for k, v in results.items():
            print(f" {k:<10}: {v:.2f}%")
        print("=" * 40)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a DML test set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (e.g., best_model_*.pth)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_SPECS),
                        help="Dataset to evaluate on")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override the default data root")
    args = parser.parse_args()

    print(f"Starting evaluation with checkpoint: {args.checkpoint}")
    evaluate_best_model(args.dataset, args.checkpoint, data_root=args.data_root)