from torchvision import datasets, transforms
from torch.utils.data import Dataset
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.criteo_preprocessing import (
    CRITEO_CATEGORY_BUCKET_SIZE,
    CRITEO_CATEGORY_COUNT,
    CRITEO_FEATURE_COUNT,
    CRITEO_FEATURE_NAMES,
    CRITEO_FEATURE_SIZES,
    CRITEO_HASH_SALT,
    CRITEO_INTEGER_COUNT,
    CRITEO_PREPROCESSING_VERSION,
    CRITEO_RAW_COLUMNS,
    normalize_integer,
    stable_category_bucket,
)

feature_sizes = []


def _validate_criteo_arrays(features, labels, source):
    if features.ndim != 2 or features.shape[1] != CRITEO_FEATURE_COUNT:
        raise ValueError(f"Invalid {source} Criteo feature shape: {features.shape}")
    if len(features) != len(labels):
        raise ValueError(f"{source} Criteo feature and label lengths do not match.")
    if not np.isfinite(features).all():
        raise ValueError(f"{source} Criteo features contain non-finite values.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{source} Criteo labels must be binary.")
    categories = features[:, CRITEO_INTEGER_COUNT:]
    if categories.size and (
        categories.min() < 0
        or categories.max() > CRITEO_CATEGORY_BUCKET_SIZE
        or not np.equal(categories, np.floor(categories)).all()
    ):
        raise ValueError(f"{source} Criteo category bucket is invalid.")


class Criteo(Dataset):
    """Load private Criteo data with the public, fit-free shared transform."""

    def __init__(self, root="./dataset", train=True, balanced=True, **kwargs):
        self.train = train
        self.root = Path(root) / "criteo"
        if not self.root.is_dir():
            raise ValueError(
                "Private Criteo train/test data is missing at {}. Download the "
                "official dac.tar.gz, then run prepare_criteo_private.py. The "
                "official test.txt is unlabeled and must not be used for accuracy "
                "evaluation.".format(self.root)
            )

        split = "train" if train else "test"
        balance_name = "balanced" if balanced else "unbalanced"
        cache_path = self.root / (
            f"{split}_processed_{CRITEO_PREPROCESSING_VERSION}_{balance_name}.npz"
        )
        if cache_path.is_file():
            self.features, self.labels = self._load_cache(cache_path)
        else:
            raw_path = self.root / f"{split}.txt"
            if not raw_path.is_file():
                raise FileNotFoundError(f"Criteo {split} file not found: {raw_path}")
            raw = pd.read_csv(
                raw_path,
                sep="\t",
                header=None,
                names=CRITEO_RAW_COLUMNS,
                nrows=1_000_000 if balanced else 60_000,
                dtype=str,
                keep_default_na=False,
            )
            if raw.shape[1] != len(CRITEO_RAW_COLUMNS):
                raise ValueError(
                    f"Expected 40 Criteo columns in {raw_path}, got {raw.shape[1]}."
                )
            raw["label"] = pd.to_numeric(raw["label"], errors="coerce")
            raw = raw[raw["label"].isin([0, 1])].reset_index(drop=True)
            if balanced:
                zeros = raw[raw["label"] == 0].head(30_000)
                ones = raw[raw["label"] == 1].head(30_000)
                pair_count = min(len(zeros), len(ones))
                if pair_count == 0:
                    raise ValueError(f"Criteo {split} has no examples from both classes.")
                order = np.empty(pair_count * 2, dtype=np.int64)
                order[0::2] = zeros.index.to_numpy()[:pair_count]
                order[1::2] = ones.index.to_numpy()[:pair_count]
                raw = raw.loc[order].reset_index(drop=True)
            if raw.empty:
                raise ValueError(f"Criteo {split} contains no valid labeled rows.")

            self.features = self._transform_features(raw)
            self.labels = raw["label"].to_numpy(dtype=np.int64)
            self._write_cache(cache_path)
            print(
                "Criteo {} preprocessing completed: rows={}, transform={}.".format(
                    split, len(self.labels), CRITEO_PREPROCESSING_VERSION
                )
            )

        _validate_criteo_arrays(self.features, self.labels, f"private {split}")

        if train:
            global feature_sizes
            feature_sizes.clear()
            feature_sizes.extend(CRITEO_FEATURE_SIZES)

    @staticmethod
    def _transform_features(raw):
        features = np.empty((len(raw), CRITEO_FEATURE_COUNT), dtype=np.float32)
        for index in range(CRITEO_INTEGER_COUNT):
            column = f"I{index + 1}"
            features[:, index] = raw[column].map(normalize_integer).to_numpy(
                dtype=np.float32
            )
        for index in range(CRITEO_CATEGORY_COUNT):
            column = f"C{index + 1}"
            features[:, CRITEO_INTEGER_COUNT + index] = raw[column].map(
                lambda value, field=index + 1: stable_category_bucket(value, field)
            ).to_numpy(dtype=np.float32)
        return features

    def _write_cache(self, cache_path):
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=self.features,
                labels=self.labels,
                preprocessing_version=np.asarray(CRITEO_PREPROCESSING_VERSION),
                hash_salt=np.asarray(CRITEO_HASH_SALT),
                bucket_size=np.asarray(CRITEO_CATEGORY_BUCKET_SIZE),
                feature_names=np.asarray(CRITEO_FEATURE_NAMES),
            )
        temporary.replace(cache_path)

    @staticmethod
    def _load_cache(cache_path):
        with np.load(cache_path, allow_pickle=False) as archive:
            version = str(archive["preprocessing_version"].item())
            salt = str(archive["hash_salt"].item())
            bucket_size = int(archive["bucket_size"].item())
            if (
                version != CRITEO_PREPROCESSING_VERSION
                or salt != CRITEO_HASH_SALT
                or bucket_size != CRITEO_CATEGORY_BUCKET_SIZE
            ):
                raise ValueError(
                    f"Stale or incompatible Criteo cache: {cache_path}. Delete it "
                    "and rerun to rebuild with the current public transform."
                )
            features = archive["features"].astype(np.float32, copy=False)
            labels = archive["labels"].astype(np.int64, copy=False)
        return features, labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], int(self.labels[idx])


class Criteo1TBPublic(Dataset):
    """Processed public Criteo 1TB shard used only as RL auxiliary data."""

    def __init__(self, root="./dataset"):
        artifact_dir = Path(root) / "criteo_1tb_public"
        npz_path = artifact_dir / "public_aux_processed.npz"
        metadata_path = artifact_dir / "public_aux_processed_metadata.json"
        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "Processed Criteo 1TB auxiliary artifacts are missing. Expected "
                f"{npz_path} and {metadata_path}. Run process_criteo_1tb_aux.py first."
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        categorical = metadata.get("categorical_transform", {})
        if (
            categorical.get("hash_salt") != CRITEO_HASH_SALT
            or int(categorical.get("bucket_size", -1))
            != CRITEO_CATEGORY_BUCKET_SIZE
            or metadata.get("feature_order") != CRITEO_FEATURE_NAMES
        ):
            raise ValueError(
                "Public Criteo auxiliary preprocessing does not match the private "
                "Criteo transform. Re-run process_criteo_1tb_aux.py."
            )
        with np.load(npz_path, allow_pickle=False) as archive:
            self.features = archive["features"].astype(np.float32, copy=False)
            self.labels = archive["labels"].astype(np.int64, copy=False)
        _validate_criteo_arrays(self.features, self.labels, "public auxiliary")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], int(self.labels[idx])


datasets_choices = [
    "mnist",
    "fashionmnist",
    "cifar10",
    "cifar100",
    "criteo"
]

datasets_name = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "criteo": "Criteo"
}

datasets_dict = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "criteo": Criteo
}

datasets_classes = {
    "mnist": 10,
    "fashionmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "criteo": 2
}

transforms_default = {
    "mnist": transforms.Compose([transforms.ToTensor()]),
    "fashionmnist": transforms.Compose([transforms.ToTensor()]),
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ]),
    "cifar100": transforms.Compose([ transforms.ToTensor()]),
    "criteo": None
}

transforms_augment = {
    "mnist": transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "fashionmnist": transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ]),
    "cifar100": transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "criteo": None
}


def build_train_transform(dataset_name, args=None):
    if dataset_name != "cifar10" or args is None:
        return transforms_default.get(dataset_name)

    disable_fia_geometry = (
        getattr(args, "attack", None) == "ressfl_fia"
        and bool(getattr(args, "fia_disable_cifar_geometric_augmentation", False))
    )
    transform_steps = []
    if not disable_fia_geometry:
        transform_steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])
    if bool(getattr(args, "cifar_random_erasing", 1)):
        transform_steps.append(
            transforms.RandomErasing(
                p=float(getattr(args, "cifar_random_erasing_p", 0.25)),
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
            )
        )
    return transforms.Compose(transform_steps)


aux_public_dataset_choices = [
    "auto",
    "none",
    "criteo_1tb",
    "emnist_letters",
    "cifar100",
    "cifar100_gray",
    "kmnist",
]


aux_public_dataset_defaults = {
    "mnist": "emnist_letters",
    "fashionmnist": "kmnist",
    "cifar10": "cifar100",
    "criteo": "criteo_1tb",
}


aux_public_dataset_names = {
    "auto": "Auto",
    "none": "None",
    "criteo_1tb": "Criteo 1TB Click Logs",
    "emnist_letters": "EMNIST Letters",
    "cifar100": "CIFAR-100",
    "cifar100_gray": "CIFAR-100 Gray 28x28",
    "kmnist": "KMNIST",
}


def resolve_aux_public_dataset_name(target_dataset, requested_name):
    if requested_name == "auto":
        return aux_public_dataset_defaults.get(target_dataset, "none")
    return requested_name


def get_aux_public_label_compatibility(target_dataset, aux_dataset_name):
    if aux_dataset_name in ["none", None]:
        return False
    if target_dataset == "criteo" and aux_dataset_name == "criteo_1tb":
        return True
    return aux_dataset_name == target_dataset


def build_aux_public_transform(target_dataset, aux_dataset_name):
    if aux_dataset_name == "emnist_letters":
        return transforms.Compose([transforms.ToTensor()])
    if aux_dataset_name == "kmnist":
        return transforms.Compose([transforms.ToTensor()])
    if aux_dataset_name == "cifar100":
        if target_dataset == "cifar10":
            return transforms_default["cifar10"]
        return transforms.Compose([transforms.ToTensor()])
    if aux_dataset_name == "cifar100_gray":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
    return None


def build_aux_public_dataset(root, target_dataset, aux_dataset_name="auto", download=True):
    resolved_name = resolve_aux_public_dataset_name(target_dataset, aux_dataset_name)
    label_compatible = get_aux_public_label_compatibility(target_dataset, resolved_name)

    if resolved_name == "none":
        return None, resolved_name, label_compatible

    if resolved_name == "criteo_1tb":
        if target_dataset != "criteo":
            raise ValueError("criteo_1tb auxiliary data only supports Criteo.")
        return Criteo1TBPublic(root=root), resolved_name, label_compatible

    transform = build_aux_public_transform(target_dataset, resolved_name)

    if resolved_name == "emnist_letters":
        dataset = datasets.EMNIST(
            root=root,
            split="letters",
            train=True,
            download=download,
            transform=transform,
        )
    elif resolved_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=root,
            train=True,
            download=download,
            transform=transform,
        )
    elif resolved_name == "cifar100_gray":
        dataset = datasets.CIFAR100(
            root=root,
            train=True,
            download=download,
            transform=transform,
        )
    elif resolved_name == "kmnist":
        dataset = datasets.KMNIST(
            root=root,
            train=True,
            download=download,
            transform=transform,
        )
    else:
        raise ValueError("Unsupported auxiliary public dataset: {}".format(resolved_name))

    return dataset, resolved_name, label_compatible
