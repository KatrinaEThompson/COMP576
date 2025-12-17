"""
model_and_data_classes.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import rasterio

TARGET_COLUMNS = [
    'AG_LIVE_CARB_ACRE', 'BG_LIVE_CARB_ACRE',
    'DEAD_WOOD_CARB_ACRE', 'LITTER_CARB_ACRE',
    'SOIL_ORG_CARB_ACRE'
]


NUM_CHANNELS = 64
IMAGE_SIZE = 161
NUM_OUTPUTS = len(TARGET_COLUMNS)
IGNORE_VALUE = -9999.0

class FIAGSEDataset(Dataset):
    def __init__(self, ground_truth_csv_path, geotiff_dir, target_cols,
                 target_means_stds=None, transform=None, ignore_value=IGNORE_VALUE):
        self.gt_df = pd.read_csv(ground_truth_csv_path)
        self.target_cols = target_cols
        self.transform = transform
        self.ignore_value = ignore_value
        self.target_means_stds = target_means_stds

        geotiff_path_lookup = {}
        for f in os.listdir(geotiff_dir):
            if f.lower().endswith(('.tif', '.tiff')):
                base_name = Path(f).stem
                parts = base_name.split('_')
                try:
                    meas_year = int(parts[-1])
                except ValueError:
                    continue
                plt_cn = '_'.join(parts[:-1])
                geotiff_path_lookup[(plt_cn, meas_year)] = os.path.join(geotiff_dir, f)

        self.samples = []
        plt_cn_column_name = self.gt_df.columns[0]
        for idx, row in self.gt_df.iterrows():
            key = (str(row[plt_cn_column_name]), int(row['MEASYEAR']))
            if key in geotiff_path_lookup:
                self.samples.append((geotiff_path_lookup[key], idx))

        print(f"Loaded {len(self.gt_df)} rows, matched {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_idx = self.samples[idx]
        with rasterio.open(img_path) as src:
            image = np.asarray(src.read())
        image_tensor = torch.from_numpy(image).float()
        if image_tensor.ndim == 3 and image_tensor.shape[0] != NUM_CHANNELS:
            if image_tensor.shape[2] == NUM_CHANNELS:
                image_tensor = image_tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected image shape {image_tensor.shape}")
        image_tensor = image_tensor[:, :IMAGE_SIZE, :IMAGE_SIZE]

        label_vector = self.gt_df.loc[gt_idx, self.target_cols].values.astype(np.float32)
        valid_mask = label_vector > 0.0
        label_tensor = label_vector.copy()
        label_tensor[~valid_mask] = self.ignore_value
        label_tensor = torch.from_numpy(label_tensor).float()

        # normalize valid entries! prevent vanishing gradients
        if self.target_means_stds is not None:
            means = self.target_means_stds['means'].numpy().astype(np.float32)
            stds = self.target_means_stds['stds'].numpy().astype(np.float32)
            for i in range(len(label_tensor)):
                if label_tensor[i] != self.ignore_value:
                    label_tensor[i] = (label_tensor[i] - means[i]) / (stds[i] if stds[i] != 0 else 1.0)

        return image_tensor, label_tensor, torch.from_numpy(valid_mask.astype(np.bool_))


class ConditionalWeightedLoss(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss(reduction='none'), weights=None, ignore_value=IGNORE_VALUE):
        super().__init__()
        self.loss_fn = loss_fn
        self.ignore_value = ignore_value
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32)
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
        self.register_buffer('_weights', weights)

    def forward(self, y_pred, y_true, mask=None):
        if mask is None:
            mask = (y_true != self.ignore_value)
        element_loss = self.loss_fn(y_pred, y_true)
        mask_f = mask.float()
        masked_loss = element_loss * mask_f
        if self._weights is not None:
            weights = self._weights.to(masked_loss.device)
            masked_loss = masked_loss * weights
        total_sum_loss = masked_loss.sum()
        num_valid = mask_f.sum()
        return total_sum_loss / num_valid if num_valid > 0 else torch.tensor(0.0, device=total_sum_loss.device)


class CNNEncoderRegressionHead(nn.Module):
    def __init__(self, num_input_channels=NUM_CHANNELS, num_output_features=NUM_OUTPUTS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.15),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(4), nn.Dropout(0.2),
        )
        final_spatial = IMAGE_SIZE // 16
        self.flattened_size = 256 * final_spatial * final_spatial
        self.regression_head = nn.Sequential(
            nn.Linear(self.flattened_size, 2048), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_output_features)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.regression_head(x)