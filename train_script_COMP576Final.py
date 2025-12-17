"""
train_script.py
"""
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score

from model_and_data_classes import (
    FIAGSEDataset, CNNEncoderRegressionHead, ConditionalWeightedLoss,
    TARGET_COLUMNS, NUM_OUTPUTS
)

# hyperparameters
batch_size = 64
epochs = 30
lr = 1e-4
try_cuda = True
seed = 1000
logging_interval = 100

# Paths 
GROUND_TRUTH_CSV = r"F:/BCarbon/rFIA/Carbon_Estimates/2017_2024_ALL_Public_byPlotTrue_byPoolTrue_CarbonCompAGBBG_reshaped_AGBGadded_filtered_ground_truth.csv"
GEOTIFF_DIR = r"F:\BCarbon\AEF\FullData\GEE_Embeddings_UTM"


# CSV for results 
datetime_str = datetime.now().strftime("%b%d_%H-%M-%S")
results_dir = Path("./results")
results_dir.mkdir(exist_ok=True)
results_csv_path = results_dir / f"{datetime_str}_training_metrics.csv"

# GPU stuff
device = torch.device("cuda" if torch.cuda.is_available() and try_cuda else "cpu")
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(seed)
print(f"Using device: {device}")

# compute normalization stats and weights
gt_df_full = pd.read_csv(GROUND_TRUTH_CSV)
means_list, stds_list = [], []

for col in TARGET_COLUMNS:
    valid_data = gt_df_full[col][gt_df_full[col] > 0].astype(float)
    m = valid_data.mean() if len(valid_data) > 0 else 0.0
    s = valid_data.std(ddof=0) if len(valid_data) > 0 and valid_data.std(ddof=0) > 0 else 1.0
    means_list.append(m)
    stds_list.append(s)

TARGET_MEANS = torch.tensor(means_list, dtype=torch.float32)
TARGET_STDS = torch.tensor(stds_list, dtype=torch.float32)

weights_for_loss = TARGET_STDS.clone()
weights_for_loss = weights_for_loss * (NUM_OUTPUTS / weights_for_loss.sum())
initial_weights = weights_for_loss.to(device)

print("\n--- Per-pool normalization stats and weights ---")
for name, m, s, w in zip(TARGET_COLUMNS, means_list, stds_list, initial_weights.cpu().numpy()):
    print(f"{name:<25} mean={m:.4f} std={s:.4f} weight={w:.6f}")
print("------------------------------------------------\n")

# tensrboard stuff
datetime_str = datetime.now().strftime("%b%d_%H-%M-%S")
logging_dir = Path(f"./runs/{datetime_str}_ForestCarbon_CNN")
logging_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(logging_dir.absolute()))

# create full dataset and get mean/std
normalization_stats = {'means': TARGET_MEANS, 'stds': TARGET_STDS}
full_dataset = FIAGSEDataset(
    ground_truth_csv_path=GROUND_TRUTH_CSV,
    geotiff_dir=GEOTIFF_DIR,
    target_cols=TARGET_COLUMNS,
    target_means_stds=normalization_stats
)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=(device.type == 'cuda'))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=(device.type == 'cuda'))

# optimizers/schedulers
model = CNNEncoderRegressionHead().to(device)

criterion = ConditionalWeightedLoss(loss_fn=nn.L1Loss(reduction='none'),
                                    weights=initial_weights, ignore_value=-9999.0)

#optimizer = optim.AdamW(model.parameters(), lr=lr) #fail
optimizer = optim.Adam(model.parameters(), lr=lr)

# reduce LR on plateau #notbad...
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

epoch_losses = []
best_global_r2 = -np.inf
best_model_path = results_dir / f"{datetime_str}_best_model.pt"

# ____________________________
# train function
# ______________________________
def train(epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (data, target, mask) in enumerate(train_loader):
        data, target, mask = data.to(device), target.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        for i, pool_name in enumerate(TARGET_COLUMNS):
            mask_i = mask[:, i].bool()
            if mask_i.any():
                pool_loss = torch.abs(output[mask_i, i] - target[mask_i, i]).mean()
                writer.add_scalar(f'train/pool_loss/{pool_name}', pool_loss.item(), epoch * num_batches + batch_idx)

        if batch_idx % logging_interval == 0:
            n_iter = epoch * num_batches + batch_idx
            print(f"Train Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")
            writer.add_scalar('train/batch_loss', loss.item(), n_iter)

    avg_epoch_loss = total_loss / max(1, num_batches)
    epoch_losses.append(avg_epoch_loss)
    writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
    print(f"--- Train Epoch {epoch} Finished. Average Loss: {avg_epoch_loss:.6f} ---")
    return avg_epoch_loss

# ____________________________
# testfunction
# ______________________________
def test(epoch):
    global best_global_r2
    model.eval()
    test_loss_total = 0.0
    test_batches = 0

    global_targets_denorm, global_preds_denorm = [], []
    global_targets_norm, global_preds_norm = [], []
    per_pool_targets = [[] for _ in range(NUM_OUTPUTS)]
    per_pool_preds = [[] for _ in range(NUM_OUTPUTS)]

    first_batch_done = False

    with torch.no_grad():
        for batch_idx, (data, target, mask) in enumerate(test_loader):
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            output = model(data)
            batch_loss = criterion(output, target, mask)
            test_loss_total += batch_loss.item()
            test_batches += 1

            means = TARGET_MEANS.to(device).unsqueeze(0)
            stds = TARGET_STDS.to(device).unsqueeze(0)
            output_denorm = output * stds + means
            target_denorm = target * stds + means

            #troubleshooting print outs - remove later
            if not first_batch_done:
                print("\n--- FIRST BATCH DIAGNOSTICS ---")
                print("Denorm target sample:", target_denorm[:5])
                print("Denorm pred sample:", output_denorm[:5])
                print("Per-pool stds:", TARGET_STDS.cpu().numpy())
                print("Loss weights:", initial_weights.cpu().numpy())
                first_batch_done = True

            valid_mask = mask.bool()

            for i in range(NUM_OUTPUTS):
                v = valid_mask[:, i]
                if v.any():
                    per_pool_targets[i].append(target_denorm[v, i].cpu().numpy())
                    per_pool_preds[i].append(output_denorm[v, i].cpu().numpy())

                    #per-pool test loss
                    pool_loss = np.mean(np.abs(output_denorm[v, i].cpu().numpy() - target_denorm[v, i].cpu().numpy()))
                    writer.add_scalar(f'test/pool_loss/{TARGET_COLUMNS[i]}', pool_loss, epoch)

            # get global stuff
            for b in range(output_denorm.shape[0]):
                v = valid_mask[b]
                if v.any():
                    global_targets_denorm.append(target_denorm[b, v].cpu().numpy())
                    global_preds_denorm.append(output_denorm[b, v].cpu().numpy())
                    global_targets_norm.append(target[b, v].cpu().numpy())
                    global_preds_norm.append(output[b, v].cpu().numpy())

    avg_test_loss = test_loss_total / max(1, test_batches)
    writer.add_scalar("test/epoch_loss", avg_test_loss, epoch)
    print(f"\n--- Test Loss (normalized Fix-A) Epoch {epoch}: {avg_test_loss:.6f} ---")

    # get global metrics
    global_targets_denorm_all = np.concatenate(global_targets_denorm)
    global_preds_denorm_all = np.concatenate(global_preds_denorm)
    global_targets_norm_all = np.concatenate(global_targets_norm)
    global_preds_norm_all = np.concatenate(global_preds_norm)

    global_mae = np.mean(np.abs(global_preds_denorm_all - global_targets_denorm_all))
    global_rmse = np.sqrt(np.mean((global_preds_denorm_all - global_targets_denorm_all) ** 2))
    global_bias = np.mean(global_preds_denorm_all - global_targets_denorm_all)
    global_r2 = r2_score(global_targets_norm_all, global_preds_norm_all)

    print("\n--- GLOBAL METRICS ---")
    print(f"R² (normalized): {global_r2:.4f}")
    print(f"RMSE: {global_rmse:.4f}")
    print(f"MAE: {global_mae:.4f}")
    print(f"Bias: {global_bias:.4f}")

    # save best model based on global R2!!!
    if global_r2 > best_global_r2:
        best_global_r2 = global_r2
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Best model updated at Epoch {epoch} with R²={global_r2:.4f} ***")

    #print per-pool metrics
    per_pool_results = {}
    print("\n--- PER-POOL METRICS ---")
    for i, pool_name in enumerate(TARGET_COLUMNS):
        if len(per_pool_targets[i]) > 0:
            t = np.concatenate(per_pool_targets[i])
            p = np.concatenate(per_pool_preds[i])
            mae = np.mean(np.abs(p - t))
            rmse = np.sqrt(np.mean((p - t)**2))
            r2 = r2_score(t, p)
            bias = np.mean(p - t)
            per_pool_results[pool_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Bias': bias}
            print(f"{pool_name:<25} MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f} Bias={bias:.4f}")
        else:
            per_pool_results[pool_name] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Bias': np.nan}

    #  save CSV
    row = {
        'Epoch': epoch,
        'Train_Loss': epoch_losses[-1],
        'Test_Loss': avg_test_loss,
        'Global_MAE': global_mae,
        'Global_RMSE': global_rmse,
        'Global_Bias': global_bias,
        'Global_R2': global_r2
    }
    for pool_name in TARGET_COLUMNS:
        for metric in ['MAE', 'RMSE', 'Bias', 'R2']:
            row[f"{pool_name}_{metric}"] = per_pool_results[pool_name][metric]

    if results_csv_path.exists():
        df_results = pd.read_csv(results_csv_path)
        df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
    else:
        df_results = pd.DataFrame([row])
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to {results_csv_path}")

    return avg_test_loss

# --- MAIN LOOP ---
#if __name__ == "__main__":
#    for epoch in range(1, epochs + 1):
#        print(f"\n--- Starting Training Epoch {epoch} ---")
#        train(epoch)
#        avg_test_loss = test(epoch)
#        scheduler.step(avg_test_loss)  # update LR if plateau detected
#    writer.close()
#    print("\nTraining finished and TensorBoard logs saved.")
