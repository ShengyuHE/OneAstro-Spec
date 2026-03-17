'''
To activate enviroment
conda activate SpecFun
'''
import os, sys  
## set hugging face to off-line
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import json
import copy
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append("..")
from utils import SpecDataLoader, SpecFeatureExtractor, SpecFeatureLoader, FeatureDataset
from helper import setup_logging
setup_logging()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Prediction task') 
    
class TargetNormalizer:
    def __init__(self, eps=1e-8):
        self.mean = None
        self.std = None
        self.eps = eps
    def fit(self, y):
        # y: (N, D)
        self.mean = y.mean(axis=0, keepdims=True)   # shape (1, D)
        self.std = y.std(axis=0, keepdims=True)     # shape (1, D)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        return self
    def transform(self, y):
        return (y - self.mean) / self.std
    def inverse_transform(self, y_norm):
        return y_norm * self.std + self.mean

class CrossAttentionPool(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True)
    def forward(self, x):
        # x: (B, T, D)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)   # (B, 1, D)
        pooled, attn_weights = self.attn(q, x, x)  # pooled: (B, 1, D)
        return pooled.squeeze(1)  # (B, D)
    
class AIONRegressor(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, output_dim=5, num_heads=8, dropout=0.1):
        super().__init__()
        self.pool = CrossAttentionPool(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
            )
    def forward(self, x):
        # x: (B, 256, 768)
        x = self.pool(x)   # (B, 768)
        x = self.mlp(x)    # (B, output_dim)
        return x

def compute_regression_metrics(y_true, y_pred, label_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    assert y_true.shape == y_pred.shape
    n_targets = y_true.shape[1]
    if label_names is None:
        label_names = [f"target_{i}" for i in range(n_targets)]
    metrics = {}
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        denom = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1.0 - np.sum((yt - yp) ** 2) / denom if denom > 0 else np.nan
        metrics[name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }
        if name == "z":
            dz_norm = (yp - yt) / (1.0 + yt)
            metrics[name]["mean_abs_dz_norm"] = float(np.mean(np.abs(dz_norm)))
            metrics[name]["std_dz_norm"] = float(np.std(dz_norm))
    return metrics

def run_epoch(model, loader, criterion, device, optimizer=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch_features.size(0)
            all_true.append(batch_labels.detach().cpu().numpy())
            all_pred.append(outputs.detach().cpu().numpy())
    total_loss /= len(loader.dataset)
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    return total_loss, all_true, all_pred

def run_direct_z(data_name, mod, config, max_samples=None):
    """
    Directly predict redshift z from the pretrained model
    """
    from aion.modalities import Z
    from contextlib import nullcontext

    def _token_to_redshift(token_probs):
        tokens = torch.arange(len(token_probs))
        expected_token = torch.sum(tokens * token_probs)
        return (expected_token / 1024.0 * 6.0).item()
    def _modalities_dict_to_list(modalities):
        modality_list = []
        if "image" in modalities:
            modality_list.append(modalities["image"])
        if "photometry" in modalities:
            phot = modalities["photometry"]
            for key in ["g", "r", "i", "z"]:
                if key in phot:
                    modality_list.append(phot[key])
        if "spectrum" in modalities:
            modality_list.append(modalities["spectrum"])
        if not modality_list:
            raise ValueError("No modalities found to encode.")
        return modality_list
    loader = SpecDataLoader(data_name)
    extractor = SpecFeatureExtractor(device=config["device"])
    data = loader.load_data(name=data_name)
    chunk_data = loader.chunk_data(batch_size=config["batch_size"],max_samples=max_samples,data=data,)
    all_results = []
    running_index = 0
    for batch_idx, batch_data in enumerate(tqdm(chunk_data, desc="Direct z inference")):
        kind = mod_to_kind[mod]
        modalities = extractor.build_modalities(batch_data, kind=kind)
        modality_list = _modalities_dict_to_list(modalities)
        labels = extractor.get_labels(batch_data, name=data_name, qu=["z"],as_tensor=False,)
        true_z = np.asarray(labels["z"]).reshape(-1)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = extractor.codec_manager.encode(*modality_list)
                preds = extractor.model(tokens, target_modality=Z)
        current_batch_size = len(batch_data)
        for k in range(current_batch_size):
            tz = float(true_z[k])
            result = {"index": running_index, "z_true": tz,}
            pred_logits = preds["tok_z"][k].squeeze()
            prob = torch.softmax(pred_logits, dim=0).detach().float().cpu()
            pred_z = float(_token_to_redshift(prob))
            dz = pred_z - tz
            dz_norm = dz / (1.0 + tz)
            result[f"z_pred"] = pred_z
            result[f"dz"] = dz
            result[f"abs_dz"] = abs(dz)
            result[f"dz_norm"] = dz_norm
            result[f"abs_dz_norm_"] = abs(dz_norm)
            all_results.append(result)
            running_index += 1
        if "cuda" in str(config["device"]):
            torch.cuda.empty_cache()
    return all_results

def run_predict_labels(features, targets, labels, config):
    """
    Train + evaluate regression model on AION features.
    """
    # dataset split
    output_dim = targets.shape[1] if targets.ndim > 1 else 1
    dataset = FeatureDataset(features, targets, task="regression")
    train_size = int(config["train_ratio"] * len(dataset))
    val_size = int(config["val_ratio"] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),)
    train_idx = train_dataset.indices
    val_idx = val_dataset.indices
    test_idx = test_dataset.indices
    # normalization (fit on train only)
    normalizer = TargetNormalizer()
    normalizer.fit(targets[train_idx])
    logger.info(f"Targets: {labels}")
    logger.info(f"Target mean: {normalizer.mean}")
    logger.info(f"Target std: {normalizer.std}")
    targets_norm = normalizer.transform(targets)
    dataset_norm = FeatureDataset(features, targets_norm, task="regression")
    train_dataset = torch.utils.data.Subset(dataset_norm, train_idx)
    val_dataset = torch.utils.data.Subset(dataset_norm, val_idx)
    test_dataset = torch.utils.data.Subset(dataset_norm, test_idx)
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,pin_memory=True,num_workers=CONFIG.get("num_workers", 4))
    val_loader = DataLoader(val_dataset,batch_size=CONFIG["batch_size"],shuffle=False,pin_memory=True,num_workers=CONFIG.get("num_workers", 4))
    test_loader = DataLoader(test_dataset,batch_size=CONFIG["batch_size"],shuffle=False,pin_memory=True,num_workers=CONFIG.get("num_workers", 4))
    # trainning model
    model = AIONRegressor(
        embed_dim=features.shape[2],   # 768
        hidden_dim=CONFIG["hidden_dim"],
        output_dim=output_dim,
        num_heads=CONFIG["num_heads"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                                    patience=CONFIG["lr_scheduler_patience"],
                                                    factor=CONFIG["lr_scheduler_factor"],)
    # training loop
    best_val_loss = np.inf
    best_model_state = None
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    
    for epoch in range(CONFIG["num_epochs"]):
        train_loss, train_true, train_pred = run_epoch(model=model,loader=train_loader,
            criterion=criterion,device=CONFIG["device"],optimizer=optimizer,)
        val_loss, val_true, val_pred = run_epoch(model=model,loader=val_loader,
            criterion=criterion,device=CONFIG["device"],optimizer=None,)
        train_true = normalizer.inverse_transform(train_true)
        train_pred = normalizer.inverse_transform(train_pred)
        val_true = normalizer.inverse_transform(val_true)
        val_pred = normalizer.inverse_transform(val_pred)
        train_metrics = compute_regression_metrics(train_true, train_pred, args.labels)
        val_metrics = compute_regression_metrics(val_true, val_pred, args.labels)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss- CONFIG["min_delta"]:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = (
                f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
            logger.info(msg)

        if epochs_no_improve >= CONFIG["stop_patience"]:
            logger.info(f"Early stopping at epoch {epoch+1} to avoid overfitting")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    ## test
    test_loss, test_true, test_pred = run_epoch(model=model,loader=test_loader,
        criterion=criterion,device=config["device"],optimizer=None,)
    test_true = normalizer.inverse_transform(test_true)
    test_pred = normalizer.inverse_transform(test_pred)
    test_metrics = compute_regression_metrics(test_true, test_pred, labels)
    logger.info("Test Metrics:")
    for name in args.labels:
        m = test_metrics[name]
        logger.info(f"{name}: MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | R2={m['r2']:.4f}")

    return {"model": model,"normalizer": normalizer,
        "train_idx": np.array(train_idx),"val_idx": np.array(val_idx),"test_idx": np.array(test_idx),
        "train_losses": np.array(train_losses),"val_losses": np.array(val_losses),
        "test_true": test_true,"test_pred": test_pred,"test_metrics": test_metrics,}

#############################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", type=str, default=["predict_labels"],choices=["predict_labels", "direct_z"], help="tasks to do")
    parser.add_argument("--data",type = str,  default='provabgs-v2', help="data(dataset)", choices=['provabgs-v2','desi-sv1'])
    parser.add_argument("--mods", nargs="+", type=str, default=["sp"], help="input modality, e.g. sp, im, ph, im+ph, sp+im, sp+im+ph")
    parser.add_argument("--labels", nargs="+", default=["z"], help="target labels, e.g. z, m_star, z_mw, t_age, sfr",)
    parser.add_argument("--quality", default=None, help="quality cut on the data",)
    parser.add_argument("--output", type=str, default=None, help="path to results",)
    parser.add_argument("--config", type=str, default=None, help="path to configuration files",)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite file")
    ## following pass to the config file or set as default
    # parser.add_argument("--device",type = str,  default='cuda', help="device", choices=['cuda','cpu'])
    # parser.add_argument("--batch_size", type=int, default=30,help="feature extraction batch size",)
    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")
    
    # default setting
    output_path = args.output or "/mnt/oss_nanhu100TB/default/zjq/results/SpecFun"
    config_fn = args.config or "./config/prediction_config.json"
    
    mod_to_kind = {
        "sp": ("desi_spectrum",),
        "im": ("legacy_image",),
        "ph": ("legacy_photometry",),
        "im+ph": ("legacy_image","legacy_photometry",),
        "sp+im": ("desi_spectrum", "legacy_image"),
        "sp+ph": ("desi_spectrum", "legacy_photometry"),
        "sp+im+ph": ("desi_spectrum", "legacy_image", "legacy_photometry"),
    }
    
    if args.quality

    for mod, quality in args.mods:
        with open(config_fn, "r") as f: 
            CONFIG = json.load(f)["datasets"][args.data]
        CONFIG["result_dir"] = os.path.join(output_path, "result", f"{args.data}")
        CONFIG["result_prefix"] = f"{args.data}_{mod}_{'_'.join(args.labels)}"
        os.makedirs(CONFIG["result_dir"], exist_ok=True)

        if "direct_z" in  args.tasks:
            logger.info(f"Running direct inference of redshift z for {args.data} with {mod}")
            all_results = run_direct_z(data_name=args.data,mod=mod,config=CONFIG,max_samples=None,)
            pred_path = os.path.join(CONFIG["result_dir"], f"{args.data}_{mod}_direct_z_test_predict.npz")
            true_z = np.array([r["z_true"] for r in all_results])
            pred_z = np.array([r[f"z_pred"] for r in all_results])
            metrics = compute_regression_metrics(true_z, pred_z, 'z')
            save_dict = {}
            for key in all_results[0].keys():
                save_dict[key] = np.array([r[key] for r in all_results])
            np.savez(pred_path, **save_dict)
            logger.info(f"Saved direct z predictions to {pred_path}")
            logger.info(
                f"[direct_z][{mod}] "
                f"MAE={metrics['z']['mae']:.4f} | "
                f"RMSE={metrics['z']['rmse']:.4f} | "
                f"mean|dz/(1+z)|={metrics['z']['mean_abs_dz_norm']:.4f} | "
                f"std(dz/(1+z))={metrics['z']['std_dz_norm']:.4f}")

        if "predict_labels" in args.tasks:
            logger.info(f"Running predict labels for {args.data} with {mod}")
            CONFIG["feature_dir"] = os.path.join(output_path, "features", f"{args.data}")
            os.makedirs(CONFIG["feature_dir"], exist_ok=True)
            FEATURE = SpecFeatureLoader(args.data)
            feature_fn = os.path.join(CONFIG["feature_dir"], f"{args.data}_{mod}_features.npz")
            features, targets = FEATURE.load_features(kind=mod_to_kind[mod], 
                                                      label_names=args.labels, label_dtype="float",
                                                      batch_size=CONFIG["batch_size"],
                                                      feature_fn=feature_fn,)
            logger.info(f"Features shape: {features.shape}")
            logger.info(f"Targets shape: {targets.shape}")
            all_results = run_predict_labels(features=features, targets=targets, labels=args.labels,config=CONFIG)
            suffix = None
            loss_path = os.path.join(CONFIG["result_dir"],f"{CONFIG['result_prefix']}_loss_history.npz")
            np.savez(loss_path, train_losses=np.array(all_results['train_losses']),val_losses=np.array(all_results['val_losses']))
            logger.info(f"Saved loss history to {loss_path}")
            model_path = os.path.join(CONFIG["result_dir"], f"{CONFIG['result_prefix']}_best.pt")
            torch.save(all_results['model'].state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
            pred_path = os.path.join(CONFIG["result_dir"],f"{CONFIG['result_prefix']}_test_predict.npz")
            np.savez(pred_path,test_true=all_results['test_true'],test_pred=all_results['test_pred'],label_names=np.array(args.labels))
            logger.info(f"Saved test predictions to {pred_path}")
        