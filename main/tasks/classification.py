'''
To activate enviroment
conda activate SpecFun
'''
import gc
import os, sys
## set hugging face to off-line
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
import copy
import itertools
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
# import seaborn as sns

sys.path.append("..")
from utils import SpecDataLoader, SpecFeatureExtractor, FeatureDataset, SpecFeatureLoader
from helper import setup_logging
from id_mask_tools import get_id_sel
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

def run_classification(features, targets, data_name, label_names, config):
    """
    Train + evaluate classification model on AION features.
    """
    targets = np.asarray(targets)
    if targets.ndim > 1 and targets.shape[1] == 1:
        targets = targets.ravel()

    label_encoder = LabelEncoder()
    label_encoder.fit(targets)
    targets_encoded = label_encoder.transform(targets)
    num_classes = len(label_encoder.classes_)
    logger.info(f"Targets: {label_names}")
    logger.info(f"Classes: {label_encoder.classes_}")
    dataset = FeatureDataset(features, targets_encoded, task="classification")
    train_size = int(config["train_ratio"] * len(dataset))
    val_size = int(config["val_ratio"] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_idx = train_dataset.indices
    val_idx = val_dataset.indices
    test_idx = test_dataset.indices
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,pin_memory=True,num_workers=config.get("num_workers", 4))
    val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=False,pin_memory=True,num_workers=config.get("num_workers", 4))
    test_loader = DataLoader(test_dataset,batch_size=config["batch_size"],shuffle=False,pin_memory=True,num_workers=config.get("num_workers", 4))
    model = AIONClassifier(
        embed_dim=features.shape[2],
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    ).to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["lr_scheduler_patience"],
        factor=config["lr_scheduler_factor"],
    )
    # training loop
    best_val_loss = np.inf
    best_model_state = None
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(config["num_epochs"]):
        train_loss, train_true, train_pred, train_prob= run_epoch(model=model,loader=train_loader,
            criterion=criterion,device=config["device"],optimizer=optimizer,)
        val_loss, val_true, val_pred, val_prob = run_epoch(model=model,loader=val_loader,
            criterion=criterion,device=config["device"],optimizer=None,)
        train_metrics = compute_classification_metrics(train_true, train_pred)
        val_metrics = compute_classification_metrics(val_true, val_pred)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss - config["min_delta"]:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = (
                f"Epoch [{epoch+1}/{config['num_epochs']}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
            )
            logger.info(msg)

        if epochs_no_improve >= config["stop_patience"]:
            logger.info(f"Early stopping at epoch {epoch+1} to avoid overfitting")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_loss, test_true, test_pred, test_prob = run_epoch(model=model,loader=test_loader,
        criterion=criterion,device=config["device"],optimizer=None,)
    test_metrics = compute_classification_metrics(test_true, test_pred)
    logger.info("Test Metrics:")
    logger.info(f"Accuracy   = {test_metrics['accuracy']:.4f}")
    logger.info(f"Macro F1   = {test_metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1= {test_metrics['weighted_f1']:.4f}")
    return {"model": model, "classes": np.array(label_encoder.classes_),
        "train_idx": np.array(train_idx),"val_idx": np.array(val_idx),"test_idx": np.array(test_idx),
        "train_losses": np.array(train_losses),"val_losses": np.array(val_losses),
        "test_true": np.array(test_true),"test_pred": np.array(test_pred),"test_prob": np.array(test_prob),"test_metrics": test_metrics,}

#####################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", type=str, default=["classification"],choices=["classification"], help="tasks to do")
    parser.add_argument("--data", type = str,  default='desi-sv1', help="dataset type", choices=['provabgs-v2','desi-sv1'])
    # parser.add_argument("--masks", nargs="+", default=[None], help="quality mask on the data",)
    parser.add_argument("--mods", nargs="+", type=str, default=["sp"], help="input modality, e.g. sp, im, ph, im+ph, sp+im, sp+im+ph")
    parser.add_argument("--labels", nargs="+", default=["type"], help="target label, e.g. type",)
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

    args.masks = [f"R_cut_{cut:.1f}" for cut in np.arange(16.5, 25., 0.2)]

    mod_to_kind = {
        "sp": ("desi_spectrum",),
        "im": ("legacy_image",),
        "ph": ("legacy_photometry",),
        "im+ph": ("legacy_image","legacy_photometry",),
        "sp+im": ("desi_spectrum", "legacy_image"),
        "sp+ph": ("desi_spectrum", "legacy_photometry"),
        "sp+im+ph": ("desi_spectrum", "legacy_image", "legacy_photometry"),
    }
    
    with open(config_fn, "r") as f:
        BASE_CONFIG = json.load(f)["datasets"][args.data]

    for mod in args.mods:
        # FEATURE loader for each modality, they will share the same id_sel and label_names
        FEATURE = SpecFeatureLoader(args.data)
        for task, mask_type in itertools.product(args.tasks, args.masks):
            CONFIG = copy.deepcopy(BASE_CONFIG)
            CONFIG["result_dir"] = os.path.join(output_path, "result", f"{args.data}", "class")
            CONFIG["feature_dir"] = os.path.join(output_path, "features", f"{args.data}")
            id_sel = None
            suffix = ""
            if mask_type is not None and mask_type != "None":
                suffix = f"_{mask_type}"
                id_sel = get_id_sel(mask_type, args.data, CONFIG)
            logger.info(f"Running classification for {args.data} with {mod}{suffix}")
            pred_path = os.path.join(CONFIG["result_dir"], f"{args.data}_{mod}_class_{'_'.join(args.labels)}{suffix}.npz")
            os.makedirs(CONFIG["result_dir"], exist_ok=True)
            os.makedirs(CONFIG["feature_dir"], exist_ok=True)
            if os.path.exists(pred_path) and not args.overwrite:
                logger.info(f"Skipping: {pred_path}")
                continue
            feature_fn = os.path.join(CONFIG["feature_dir"], f"features_{args.data}_{mod}.npz")
            features, targets = FEATURE.load_features(kind=mod_to_kind[mod], id_sel=id_sel,
                                                      label_names=args.labels, label_dtype="object",
                                                      batch_size=CONFIG["batch_size"],
                                                      feature_fn=feature_fn,)
            logger.info(f"Features shape: {features.shape}")
            logger.info(f"Targets shape: {targets.shape}")
            all_results = run_classification(features=features, targets=targets, data_name = args.data, label_names=args.labels, config=CONFIG)
            # Save results
            # loss_path = os.path.join(CONFIG["result_dir"],
                                    # f"{CONFIG['result_prefix']}_loss_history.npz")
            # np.savez(loss_path, train_losses=np.array(train_losses),val_losses=np.array(val_losses))
            # logger.info(f"Saved loss history to {loss_path}")
            # model_path = os.path.join(CONFIG["result_dir"],f"{CONFIG['result_prefix']}_best.pt")
            # torch.save(all_results["model"].state_dict(), model_path)
            # logger.info(f"Saved best model to {model_path}")
            np.savez(pred_path, classes = all_results["classes"], 
                     metrics=all_results["test_metrics"],test_true=all_results["test_true"],test_pred=all_results["test_pred"],test_prob=all_results["test_prob"],
                     label_names=np.array(args.labels), feature_shape=np.array(features.shape), target_shape=np.array(targets.shape))
            logger.info(f"Saved test predictions to {pred_path}")
        del FEATURE
        gc.collect()
        torch.cuda.empty_cache()