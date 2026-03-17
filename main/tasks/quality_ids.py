import os, sys
import tqdm
import json
import logging
import argparse
import numpy as np
from astropy.table import Table

sys.path.append("..")
from utils import SpecDataLoader
from helper import setup_logging
import id_mask_tools as IDmask

setup_logging()
logger = logging.getLogger("quality_mask")

#############################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", type=str, default=["snr_quality"],choices=["snr_quality"], help="tasks to do")
    parser.add_argument("--data",type = str,  default='provabgs-v2', help="data(dataset)", choices=['provabgs-v2','desi-sv1'])
    parser.add_argument("--output", type=str, default=None, help="path to results",)
    parser.add_argument("--config", type=str, default=None, help="path to configuration files",)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite file")
    ## following pass to the config file or set as default
    # parser.add_argument("--device",type = str,  default='cuda', help="device", choices=['cuda','cpu'])
    # parser.add_argument("--batch_size", type=int, default=30,help="feature extraction batch size",)
    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")

    output_path = args.output or "/mnt/oss_nanhu100TB/default/zjq/results/SpecFun"
    config_fn = args.config or "./config/prediction_config.json"

    with open(config_fn, "r") as f: 
        CONFIG = json.load(f)["datasets"][args.data]
    CONFIG["result_dir"] = os.path.join(output_path, "result", f"id_mask")
    os.makedirs(CONFIG["result_dir"], exist_ok=True)

    loader = SpecDataLoader(args.data)
    data = loader.load_data()
    batches = loader.chunk_data(batch_size=CONFIG["batch_size"], data=data)

    if 'snr_quality' in args.tasks:
        methods = ["mean", "median","weighted"]
        snr_cuts = [1, 2, 3, 4]
        all_mask_ids = {}
        for method in methods:
            for snr_cut in snr_cuts:
                key = f"spec_snr_{method}_cut{snr_cut}"
                all_mask_ids[key] = []
        # loop over batches
        for i, batch in enumerate(tqdm.tqdm(batches, desc="Extract mask id", dynamic_ncols=True)):
            for method in methods:
                for snr_cut in snr_cuts:
                    key = f"spec_snr_{method}_cut{snr_cut}"
                    batch_mask_id = IDmask.id_mask_spec_snr(batch, snr_cut, method=method)
                    all_mask_ids[key].append(batch_mask_id)
        # merge all batches for each mask
        final_mask_ids = {}
        for key, id_list in all_mask_ids.items():
            if len(id_list) == 0:
                final_ids = np.array([], dtype=np.int64)
            else:
                final_ids = np.concatenate(id_list)
                final_ids = np.unique(final_ids)
            final_mask_ids[key] = final_ids
            logger.info(f"{key}: {len(final_ids)} ids")    
        # save all masks into one npz file
        save_fn = os.path.join(CONFIG["result_dir"], f"ids_{args.data}_snr_quality.npz")
        if os.path.exists(save_fn) and not args.overwrite:
            raise FileExistsError(f"{save_fn} already exists. Use --overwrite to replace it.")
        np.savez(save_fn, **final_mask_ids)
        logger.info(f"Saved all masks to {save_fn}")





