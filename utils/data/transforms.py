import json

import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, roi_margin=8):
        self.isforward = isforward
        self.max_key = max_key
        self.roi_margin = int(roi_margin)

    def _roi_mask(self, target, attrs, slice_index):
        """Build a soft lesion ROI mask from fastMRI+ annotations."""
        height, width = target.shape[-2:]
        roi = np.zeros((height, width), dtype=np.float32)
        raw = attrs.get("annotations", "{}") if isinstance(attrs, dict) else "{}"
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            annotations = json.loads(raw or "{}")
        except (TypeError, ValueError):
            annotations = {}

        for box in annotations.get(str(int(slice_index)), []):
            x0 = max(0, int(box.get("x", 0)))
            y0 = max(0, int(box.get("y", 0)))
            x1 = min(width, x0 + max(0, int(box.get("width", 0))))
            y1 = min(height, y0 + max(0, int(box.get("height", 0))))
            if x1 <= x0 or y1 <= y0:
                continue

            mx0 = max(0, x0 - self.roi_margin)
            my0 = max(0, y0 - self.roi_margin)
            mx1 = min(width, x1 + self.roi_margin)
            my1 = min(height, y1 + self.roi_margin)
            roi[my0:my1, mx0:mx1] = np.maximum(
                roi[my0:my1, mx0:mx1], 0.25
            )
            roi[y0:y1, x0:x1] = 1.0

        return torch.from_numpy(roi)

    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(
            mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)
        ).byte()
        if self.isforward:
            return mask, kspace, target, maximum, fname, slice

        roi_mask = self._roi_mask(target, attrs, slice)
        return mask, kspace, target, maximum, fname, slice, roi_mask
