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
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
# import seaborn as sns

sys.path.append("..")
from utils import SpecDataLoader, SpecFeatureExtractor, FeatureDataset, SpecFeatureLoader
from helper import setup_logging
setup_logging()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Classification task")

class LabelEncoder:
    """
    Encode arbitrary labels (int/string/etc.) into [0, ..., C-1]
    """
    def __init__(self):
        self.classes_ = None
        self.class_to_idx = None
    def fit(self, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return self
    def transform(self, y):
        y = np.asarray(y)
        return np.array([self.class_to_idx[v] for v in y], dtype=np.int64)
    def inverse_transform(self, y_idx):
        y_idx = np.asarray(y_idx, dtype=np.int64)
        return self.classes_[y_idx]
    
class CrossAttentionPool(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        # x: (B, T, D)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)   # (B, 1, D)
        pooled, attn_weights = self.attn(q, x, x)
        return pooled.squeeze(1)            # (B, D)

class AIONClassifier(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, num_classes=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.pool = CrossAttentionPool(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.pool(x)      # (B, D)
        logits = self.mlp(x)  # (B, C)
        return logits
    
def compute_classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            total_loss += loss.item() * batch_features.size(0)
            all_true.append(batch_labels.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())
            all_prob.append(probs.detach().cpu().numpy())
    total_loss /= len(loader.dataset)
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_prob = np.concatenate(all_prob, axis=0)
    return total_loss, all_true, all_pred, all_prob

#####################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", type=str, default=["classfication"],choices=["classfication"], help="tasks")
    parser.add_argument("--data",type = str,  default='provabgs', help="dataset", choices=['provabgs','desi-sv1'])
    parser.add_argument("--mods", nargs="+", type=str, default=["sp"], help="input modality, e.g. sp, im, ph, im+ph, sp+im, sp+im+ph")
    parser.add_argument("--labels", nargs="+", default=["z"], help="target labels, e.g. type,  possible (z, m_star, z_mw, t_age, sfr")
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
    use_saved_feature = False
    
    mod_to_kind = {
        "sp": ("desi_spectrum",),
        "im": ("legacy_image",),
        "ph": ("legacy_photometry",),
        "im+ph": ("legacy_image","legacy_photometry",),
        "sp+im": ("desi_spectrum", "legacy_image"),
        "sp+ph": ("desi_spectrum", "legacy_photometry"),
        "sp+im+ph": ("desi_spectrum", "legacy_image", "legacy_photometry"),
    }
    

    for mod in args.mods:
        with open(config_fn, "r") as f: 
            CONFIG = json.load(f)["datasets"][args.data]
        CONFIG["result_dir"] = os.path.join(output_path, "result", f"{args.data}")
        CONFIG["result_prefix"] = f"{args.data}_{mod}_{'_'.join(args.labels)}"
        os.makedirs(CONFIG["result_dir"], exist_ok=True)

        
        if "classfication" in args.tasks:
            logger.info(f"Running classification for {args.data} with {mod}")
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
            
            # Handle single or multiple labels for classification
            if targets.ndim > 1 and targets.shape[1] == 1:
                targets = targets.ravel()
            
            # Encode labels to [0, ..., C-1]
            label_encoder = LabelEncoder()
            label_encoder.fit(targets)
            targets_encoded = label_encoder.transform(targets)
            num_classes = len(label_encoder.classes_)
            
            logger.info(f"Number of classes: {num_classes}")
            logger.info(f"Classes: {label_encoder.classes_}")
            
            # Create dataset with encoded labels
            dataset = FeatureDataset(features, targets_encoded, task="classification")
            
            # Train/val/test split
            train_size = int(CONFIG["train_ratio"] * len(dataset))
            val_size = int(CONFIG["val_ratio"] * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
            
            train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], 
                                      shuffle=True, pin_memory=True,
                                      num_workers=CONFIG.get("num_workers", 4))
            val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                                    shuffle=False, pin_memory=True,
                                    num_workers=CONFIG.get("num_workers", 4))
            test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                                     shuffle=False, pin_memory=True,
                                     num_workers=CONFIG.get("num_workers", 4))
            
            # Initialize model
            model = AIONClassifier(
                embed_dim=features.shape[2],   # 768
                hidden_dim=CONFIG["hidden_dim"],
                num_classes=num_classes,
                num_heads=CONFIG["num_heads"],
                dropout=CONFIG["dropout"],
            ).to(CONFIG["device"])
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                            patience=CONFIG["lr_scheduler_patience"],
                                                            factor=CONFIG["lr_scheduler_factor"])
            
            best_val_loss = np.inf
            best_model_state = None
            epochs_no_improve = 0
            
            train_losses = []
            val_losses = []
            
            # Training loop
            for epoch in range(CONFIG["num_epochs"]):
                train_loss, train_true, train_pred, train_prob = run_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    device=CONFIG["device"],
                    optimizer=optimizer,
                )
                
                val_loss, val_true, val_pred, val_prob = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=CONFIG["device"],
                    optimizer=None,
                )
                
                train_metrics = compute_classification_metrics(train_true, train_pred)
                val_metrics = compute_classification_metrics(val_true, val_pred)
                
                scheduler.step(val_loss)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss - CONFIG["min_delta"]:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    msg = (
                        f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] "
                        f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
                    )
                    logger.info(msg)
                
                if epochs_no_improve >= CONFIG["stop_patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1} to avoid overfitting")
                    break
            
            # Load best model and evaluate on test set
            model.load_state_dict(best_model_state)
            
            test_loss, test_true, test_pred, test_prob = run_epoch(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=CONFIG["device"],
                optimizer=None,
            )
            
            test_metrics = compute_classification_metrics(test_true, test_pred)
            
            logger.info("=" * 60)
            logger.info("Test Metrics:")
            logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Macro F1: {test_metrics['macro_f1']:.4f}")
            logger.info(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")
            # logger.info("Confusion Matrix:")
            # cm = np.array(test_metrics['confusion_matrix'])
            # logger.info(str(cm))
            # logger.info("=" * 60)
            
            # Save results
            loss_path = os.path.join(CONFIG["result_dir"],
                                    f"{CONFIG['result_prefix']}_loss_history.npz")
            np.savez(loss_path, train_losses=np.array(train_losses),val_losses=np.array(val_losses))
            logger.info(f"Saved loss history to {loss_path}")
            model_path = os.path.join(CONFIG["result_dir"],f"{CONFIG['result_prefix']}_best.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
            pred_path = os.path.join(CONFIG["result_dir"],f"{CONFIG['result_prefix']}_test_predictions.npz")
            np.savez(pred_path,test_true=test_true,test_pred=test_pred,test_prob=test_prob,
                     label_names=np.array(args.labels),
                     classes=label_encoder.classes_,metrics=test_metrics)
            logger.info(f"Saved test predictions to {pred_path}")