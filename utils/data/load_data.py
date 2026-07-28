import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import numpy as np

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def _sample_dataset(dataset, sample_rate, seed):
    if sample_rate is None or sample_rate >= 1.0:
        return dataset
    if sample_rate <= 0.0:
        raise ValueError("sample_rate should be greater than 0 and less than or equal to 1.")

    num_examples = len(dataset)
    num_samples = max(1, int(num_examples * sample_rate))
    rng = random.Random(seed)
    indices = list(range(num_examples))
    rng.shuffle(indices)
    return Subset(dataset, sorted(indices[:num_samples]))


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(
            isforward,
            max_key_,
            roi_margin=getattr(args, "roi_margin", 8),
        ),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )
    if not isforward:
        if shuffle:
            sample_rate = getattr(args, "train_sample_rate", 1.0)
            subset_seed = getattr(args, "seed", 0)
        else:
            sample_rate = getattr(args, "val_sample_rate", 1.0)
            subset_seed = getattr(args, "seed", 0) + 1
        data_storage = _sample_dataset(data_storage, sample_rate, subset_seed)
        print(f"Using {len(data_storage)} examples from {data_path}")

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
