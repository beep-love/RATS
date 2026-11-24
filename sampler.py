import time
import torch
from torch.utils.data import Sampler, DataLoader
from collections import defaultdict
import random
from cub_200_dataset import build_cub_dataset

class PKSampler(Sampler):
    """
    P x K sampler for metric learning.
    Each batch has:
    - P identities (classes)
    - K samples per identity

    So batch size = P * K.

    labels are dense class indices, e.g. 0,1,2,...,C-1
    """
    
    def __init__(self, labels, P, K, seed=33):
        """
        labels: list or 1D tensor of class indices
        P: number of identities per batch
        K: number of samples per identity
        """
        # Ensure labels is a tensor of ints
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = labels.clone().detach().long()

        self.labels = labels
        self.P = P
        self.K = K
        self.rng = random.Random(seed)

        # Build a mapping: class_id -> list of sample indices
        # Example:
        #   class 0 -> [0, 5, 8, 12]
        #   class 1 -> [1, 9, 11]

        self.index_by_class = defaultdict(list)
        for idx, c in enumerate(self.labels.tolist()):
            self.index_by_class[c].append(idx)


        # For each class, create a shuffled pool of indices
        self.class_pools = {}

        # Shuffle indices within each class pool
        for c, idxs in self.index_by_class.items():
            pool = idxs.copy()
            self.rng.shuffle(pool)
            self.class_pools[c] = pool

        # Only keep classes that have at least K samples
        self.classes = [c for c, pool in self.class_pools.items() if len(pool) >= K]

        # Precompute how many full batches are possible if we fully consume pools
        # This is a rough upper bound; actual may be smaller at runtime.
        total_possible_samples = sum(len(pool) // K for pool in self.class_pools.values()) * K
        self.max_batches = total_possible_samples // (P * K)

    def __iter__(self):
        """
        Yield indices for one epoch.

        Logic:
        - Make a local copy of class pools and shuffle it
        - While there are at least P classes left:
            * sample P classes
            * for each class, sample K indices
        - Yield exactly P*K indices per batch

        - Each sample index is used at most once per epoch.
        - Each class appears in as many P*K batches as its pool allows.
        - If a class has many images, it appears in multiple batches.
        - If a class has fewer than K images, it never appears (cannot fill K slots).
        """

        # Local copies of pools, so each epoch gets fresh consumption
        class_pools = {c: pool.copy() for c, pool in self.class_pools.items()}

        # List of classes that still have at least K remaining samples
        active_classes = [c for c, pool in class_pools.items() if len(pool) >= self.K]
        self.rng.shuffle(active_classes)

        num_batches = 0
        while True:
            # If not enough classes left, stop
            if len(active_classes) < self.P:
                break

            # Option 1: take next P classes in order
            # Option 2: random.sample from active_classes
            # I will use option 1 for deterministic consumption
            batch_classes = active_classes[:self.P]

            batch_indices = []
            classes_to_drop = []

            for c in batch_classes:
                pool = class_pools[c]
                # Take first K indices from this class pool
                chosen = pool[:self.K]
                batch_indices.extend(chosen)
                # Remove them from the pool
                class_pools[c] = pool[self.K:]

                # If this class has fewer than K remaining, drop from active list
                if len(class_pools[c]) < self.K:
                    classes_to_drop.append(c)

            # Drop depleted classes from active_classes
            active_classes = [c for c in active_classes if c not in classes_to_drop]

            # Yield the batch
            for idx in batch_indices:
                yield idx

            num_batches += 1
            if num_batches >= self.max_batches:
                break

    def __len__(self):
        # Approximate number of samples per epoch: max_batches * P*K
        return self.max_batches * self.P * self.K

def _benchmark_sampler():
    """
    Benchmark:
      - time to fetch one batch    
      - test the structure of the batch
      - time to iterate all batches in an epoch
    """

    data_root: str = None
    # dataset: "cub_200" - use an assert asking to define the path here
    assert data_root is not None, "Please define the CUB-200 dataset root path"
    
    print("Building train and val datasets...")

    train_ds, val_ds = build_cub_dataset(
        data_root, 
        image_size=256, 
        crop_size=224, 
        seed=33
    )

    labels = train_ds.get_labels()

    P = 32
    K = 8
    batch_size = P * K

    sampler = PKSampler(labels, P=P, K=K)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=24,    # increase for speed, test different values
        pin_memory=True,   # if using GPU
    )

    print("\n=== Benchmark: Fetch first batch ===")
    t0 = time.time()
    images, labels = next(iter(train_loader))
    t1 = time.time()
    print(f"Time for first batch: {t1 - t0:.4f} sec")
    print("First batch details:")
    print(f"Batch shape: {images.shape}")                   # should be [P*K, 3, H, W]
    print(f"Unique ids in batch: {labels.unique().tolist()}")
    
    # Reshape labels to [P, K] to see structure
    labels_pk = labels.view(P, K)
    print("Labels reshaped to [P, K]:")
    print(labels_pk)

    print("Unique ids per row:")
    for i in range(P):
        unique_ids = labels_pk[i].unique().tolist()
        print(f"Row {i}: {unique_ids}")

    print("\n=== Benchmark: Full epoch ===")
    t0 = time.time()
    num_batches = 0
    for batch in train_loader:
        num_batches += 1
    t1 = time.time()

    print(f"Total batches this epoch: {num_batches}")
    print(f"Time for loading full epoch: {t1 - t0:.2f} sec")
    print(f"Avg time per batch: {(t1 - t0) / num_batches:.4f} sec")


if __name__ == "__main__":
    _benchmark_sampler()