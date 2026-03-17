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

sys.path.append("..")
from utils import SpecFeatureLoader, _decode_array
from helper import setup_logging
setup_logging()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Feaures') 


#############################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="+", type=str, default=["extract_feature"],choices=["extract_feature", "update_labels"], help="tasks")
    parser.add_argument("--data",type = str,  default='provabgs-v2', help="dataset", choices=['provabgs-v2','desi-sv1'])
    parser.add_argument("--mods", nargs="+", type=str, default=["sp"], help="input modality, e.g. sp, im, ph, im+ph, sp+im, sp+im+ph")
    parser.add_argument("--labels", nargs="+", help="target labels, e.g. id, z, m_star, z_mw, t_age, sfr",)
    parser.add_argument("--output", type=str, default=None, help="path to results",)
    parser.add_argument("--config", type=str, default=None, help="path to configuration files",)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite file")
    ## following pass to the config file or set as default
    # parser.add_argument("--device",type = str,  default='cuda', help="device", choices=['cuda','cpu'])
    # parser.add_argument("--batch_size", type=int, default=30,help="feature extraction batch size",)
    # parser.add_argument("--config", type=str, default="./config/prediction_config.json", help="path to config file")
    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")
    
    # default path
    output_path = args.output or "/mnt/oss_nanhu100TB/default/zjq/results/SpecFun"
    config_fn = args.config or "./config/prediction_config.json"
    use_saved_feature = True
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
        CONFIG["feature_dir"] = os.path.join(output_path, "features", f"{args.data}")
        os.makedirs(CONFIG["feature_dir"], exist_ok=True)
        feature_fn = os.path.join(CONFIG["feature_dir"], f"{args.data}_{mod}_features.npz")
        FEATURE = SpecFeatureLoader(dataset=args.data)
        if "extract_feature" in args.task:
            if (not os.path.exists(feature_fn)) or args.overwrite:
                features, targets = FEATURE.load_features(kind=mod_to_kind[mod], 
                                                          label_names=None, label_dtype="auto",
                                                          batch_size=CONFIG["batch_size"],
                                                          feature_fn=feature_fn, 
                                                          overwrite=args.overwrite)
            logger.info(f"Feature file in: {feature_fn}")
        if "update_labels" in args.task:
            if args.labels is None:
                raise ValueError("--labels should be provided when using update_labels")
            FEATURE.update_feature_labels(feature_fn=feature_fn,
                                          label_names=args.labels, 
                                          saving=True)
            
        