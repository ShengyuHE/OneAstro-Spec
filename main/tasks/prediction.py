'''
To activate enviroment
conda activate SpecFun
'''
import os, sys  
## set hugging face to off-line
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import logging
import copy
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append("..")
from utils import SpecDataLoader, SpecFeatureExtractor
from helper import setup_logging
setup_logging()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Prediction task') 

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
    
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        if self.labels.ndim == 1:
            self.labels = self.labels.unsqueeze(-1)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def load_features(loader, extractor,
                  dataset="provabgs", kind=("desi_spectrum",), label_names=("z"),
                  batch_size=30, max_samples=21870, save_path=None, overwrite=False):
    # data = loader.load_data(name=dataset)
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        logger.info(f"Loading cached features from {save_path}")
        cache = np.load(save_path)
        features = cache["features"]
        all_labels = cache["labels"]
        all_label_names = list(cache["label_names"])
        col_idx = [all_label_names.index(name) for name in label_names]
        labels = all_labels[:, col_idx]
        return features, labels

    batches = loader.chunk_data(batch_size=batch_size, max_samples=max_samples)
    all_features = []
    all_labels = []
    all_label_names = None
    for i, batch in enumerate(tqdm(batches, desc="Feature Extraction")):
        modalities = extractor.build_modalities(batch, kind=kind)
        features = extractor.extract_features(modalities, flatten=True)
        labels_dict = extractor.get_labels(batch, name=dataset, qu=None, as_tensor=False)
        if all_label_names is None:
            all_label_names = list(labels_dict.keys())
            # print("all_label_names:", all_label_names)
        all_features.append(features.numpy().astype(np.float32))
        label_array = np.column_stack([np.asarray(labels_dict[name], dtype=np.float32) for name in all_label_names])
        all_labels.append(label_array)
        if i % 10 == 0:
            torch.cuda.empty_cache()
    all_features = np.vstack(all_features)
    all_labels = np.vstack(all_labels)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, features=all_features, labels=all_labels, label_names=np.array(all_label_names))
    col_idx = [all_label_names.index(name) for name in label_names]
    labels = all_labels[:, col_idx]
    return all_features, labels

def compute_regression_metrics(y_true, y_pred, label_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
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

def train_mlp_regressor(features, labels, label_names, CONFIG):
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    logger.info(": Training prediction task with MLP Regressor")
    dataset = FeatureDataset(features, labels)
    train_size = int(CONFIG["train_ratio"] * len(dataset))
    val_size = int(CONFIG["val_ratio"] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    input_dim = features.shape[1]
    output_dim = len(label_names)
    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=CONFIG["mlp_hidden_dims"],
        output_dim=output_dim,
        dropout=CONFIG.get("dropout", 0.3),
    ).to(CONFIG["device"])
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    train_losses = []
    val_losses = []
    train_metrics_hist = []
    val_metrics_hist = []

    best_val_loss = np.inf
    best_model_state = None
    
    for epoch in tqdm(range(CONFIG["num_epochs"]), desc="Training"):
        # Train
        model.train()
        train_loss = 0.0
        train_true = []
        train_pred = []
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(CONFIG["device"])
            batch_labels = batch_labels.to(CONFIG["device"])
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_true.append(batch_labels.detach().cpu().numpy())
            train_pred.append(outputs.detach().cpu().numpy())
        train_loss /= len(train_loader)
        train_true = np.concatenate(train_true, axis=0)
        train_pred = np.concatenate(train_pred, axis=0)
        train_metrics = compute_regression_metrics(train_true, train_pred, label_names)
        # VALIDATE
        model.eval()
        val_loss = 0.0
        val_true = []
        val_pred = []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(CONFIG["device"])
                batch_labels = batch_labels.to(CONFIG["device"])
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                val_true.append(batch_labels.detach().cpu().numpy())
                val_pred.append(outputs.detach().cpu().numpy())
        val_loss /= len(val_loader)
        val_true = np.concatenate(val_true, axis=0)
        val_pred = np.concatenate(val_pred, axis=0)
        val_metrics = compute_regression_metrics(val_true, val_pred, label_names)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_hist.append(train_metrics)
        val_metrics_hist.append(val_metrics)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 5 == 0:
            msg = (
                f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
            for name in label_names:
                msg += (
                    f" | Val {name} MAE: {val_metrics[name]['mae']:.4f}"
                    f" | Val {name} RMSE: {val_metrics[name]['rmse']:.4f}")
            print(msg)
            
    model.load_state_dict(best_model_state)
    logger.info(f"\nBest Validation Loss: {best_val_loss:.4f}")
    # TEST
    model.eval()
    test_true = []
    test_pred = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(CONFIG["device"])
            outputs = model(batch_features)
            test_true.append(batch_labels.numpy())
            test_pred.append(outputs.detach().cpu().numpy())
    test_true = np.concatenate(test_true, axis=0)
    test_pred = np.concatenate(test_pred, axis=0)
    test_metrics = compute_regression_metrics(test_true, test_pred, label_names)
    logger.info("\nFinal Test Metrics:")
    for name in label_names:
        m = test_metrics[name]
        print(
            f"  {name:<8} | MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | R2: {m['r2']:.4f}"
        )
        if name == "z":
            print(
                f" | mean|dz_norm|: {m['mean_abs_dz_norm']:.4f} | std dz_norm: {m['std_dz_norm']:.4f}"
                )
    model_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['result_prefix']}_best.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"\nBest model saved to {model_path}")
    return (model, test_loader, train_losses,val_losses,
        train_metrics_hist, val_metrics_hist, test_metrics)

#############################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type = str,  default='cuda', help="device", choices=['cuda','cpt'])
    parser.add_argument("--data",type = str,  default='provabgs', help="dataset", choices=['provabgs','desi-sv1'])
    parser.add_argument("--mod", type=str, default="sp", choices=["sp", "im", "ph", "sp+im", "sp+im+ph"], help="input modality combination")
    parser.add_argument("--labels", nargs="+", default=["z", "m_star"], help="target labels, e.g. z, m_star, z_mw, t_age, sfr",)
    parser.add_argument("--batch_size", type=int, default=30,help="feature extraction batch size",)
    parser.add_argument("--output_path", type=str, default="/mnt/oss_nanhu100TB/default/zjq/results/SpecFun", help="path to results",)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite file")
    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")
    mod = args.mod
    output_path = args.output_path
    mod_to_kind = {
        "sp": ("desi_spectrum",),
        "im": ("legacy_image",),
        "ph": ("legacy_photometry",),
        "sp+im": ("desi_spectrum", "legacy_image"),
        "sp+im+ph": ("desi_spectrum", "legacy_image", "legacy_photometry"),
    }
    loader = SpecDataLoader(args.data)
    extractor = SpecFeatureExtractor(device=args.device)
    feature_path = output_path+f"/features/provabgs_{mod}_features.npz"
    features, labels = load_features(loader=loader, extractor=extractor,
                                     dataset="provabgs", kind=mod_to_kind[mod], label_names=args.labels, batch_size=30,
                                     save_path=feature_path, overwrite=args.overwrite)
    
    

    