import os, sys 
import logging 
import argparse
import numpy as np
from astropy.table import Table

sys.path.append("..")
from helper import _decode_array, setup_logging
setup_logging()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mask_tools') 


def combine_id_masks(*id_lists):
    if len(id_lists) == 0:
        return np.array([], dtype=np.int64)
    common = set(np.asarray(id_lists[0]))
    for ids in id_lists[1:]:
        common &= set(np.asarray(ids))
    return np.array(sorted(common), dtype=np.int64)

def make_id_mask(ids, id_sel=None):
    if id_sel is None:
        return np.ones(len(ids), dtype=bool)
    ids = np.asarray(ids)
    id_sel = np.asarray(id_sel)

    # Fast path for unicode strings
    if ids.dtype.kind == "U":
        if id_sel.dtype.kind != "U":
            id_sel = id_sel.astype(ids.dtype, copy=False)
        return np.isin(ids, id_sel)

    # Bytes -> decode once
    if ids.dtype.kind == "S":
        ids = ids.astype(str)
        if id_sel.dtype.kind == "S":
            id_sel = id_sel.astype(str)
        elif id_sel.dtype.kind != "U":
            id_sel = id_sel.astype(str)
        return np.isin(ids, id_sel)
    
    # Object fallback
    if ids.dtype == object or id_sel.dtype == object:
        ids = _decode_array(ids)
        id_sel = _decode_array(id_sel)
        return np.isin(ids, id_sel)

    # Generic fallback
    if id_sel.dtype != ids.dtype:
        try:
            id_sel = id_sel.astype(ids.dtype, copy=False)
        except Exception:
            ids = ids.astype(str)
            id_sel = id_sel.astype(str)
    return np.isin(ids, id_sel)

def id_mask_z_range(data, zmin=None, zmax=None, z_col="Z", id_col="TARGETID"):
    if z_col not in data.colnames or id_col not in data.colnames:
        return np.array([], dtype=np.int64)
    z = np.asarray(data[z_col])
    mask = np.ones(len(data), dtype=bool)
    if zmin is not None:
        mask &= (z >= zmin)
    if zmax is not None:
        mask &= (z <= zmax)
    return _decode_array(data[id_col][mask])

def id_mask_tsnr2(data, snr_cut, snr_col="TSNR2_LRG", id_col="TARGETID"):
    if snr_col not in data.colnames or id_col not in data.colnames:
        return np.array([], dtype=np.int64)
    snr = np.asarray(data[snr_col])
    mask = np.isfinite(snr) & (snr >= snr_cut)
    return _decode_array(data[id_col][mask])

def id_mask_zquality(data, zq_cut, zq_col="Z_Quality", id_col="TARGETID"):
    if zq_col not in data.colnames or id_col not in data.colnames:
        return np.array([], dtype=np.int64)
    zq = np.asarray(data[zq_col])
    mask = np.isfinite(zq) & (zq >= zq_cut)
    return _decode_array(data[id_col][mask])

def id_mask_spec_snr(data, snr_cut, method="quadratic", id_col="TARGETID"):
    flux = np.asarray(data["desi_spectrum_flux"])
    ivar = np.asarray(data["desi_spectrum_ivar"])
    mask_spec = np.asarray(data["desi_spectrum_mask"])
    valid = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (mask_spec == 0)
    n_valid = np.sum(valid, axis=1)
    snr = np.full(len(data), np.nan, dtype=float)
    snr_pixel = flux * np.sqrt(ivar)
    snr_pixel = np.where(valid, snr_pixel, np.nan)
    good = n_valid > 0
    if method == "mean":
        snr[good] = np.nanmean(snr_pixel[good], axis=1)
    elif method == "median":
        snr[good] = np.nanmedian(snr_pixel[good], axis=1)
    elif method == "weighted":
        num = np.sum(flux * ivar * valid, axis=1)
        den = np.sqrt(np.sum(ivar * valid, axis=1))
        good = den > 0
        snr[good] = num[good] / den[good]
        snr = snr / np.median(snr)*2
    else:
        raise ValueError(f"Unknown method: {method}")
    mask = np.isfinite(snr) & (snr >= snr_cut)
    return _decode_array(data[id_col][mask])

def id_mask_magnitude(data, mag_cut, band='R', id_col="TARGETID"):
    band_col = 'MAG_' + band
    if band_col not in data.colnames or id_col not in data.colnames:
        return np.array([], dtype=np.int64)
    mag = np.asarray(data[band_col])
    mask = mag <= mag_cut
    return _decode_array(data[id_col][mask])

def get_id_masks(data_type, mask_type, id_dir = '/mnt/oss_nanhu100TB/default/zjq/results/SpecFun/result/id_mask', **args):
    fn = os.path.join(id_dir, f'ids_{data_type}_{mask_type}.npz')
    logger.info(f"Loading id masks from {fn}")
    all_id_mask = np.load(fn, allow_pickle=True)
    if mask_type == 'spec_snr':
        id_mask = all_id_mask[f'{mask_type}_{args["method"]}_cut{args["cut"]}']
        return id_mask
    elif mask_type == 'mag_cut':
        id_mask = all_id_mask[f'mag_{args["band"]}_cut_{args["cut"]}']
        return id_mask
    elif mask_type == 'R_cut':
        id_mask = all_id_mask[f'R_cut_{args["cut"]}']
        return id_mask
    elif mask_type == 'zq_cut':
        id_mask = all_id_mask[f'{mask_type}{args["cut"]}']
        return id_mask
    
def get_id_sel(mask_type, data_type, config):
    if "spec_snr" in mask_type:
        config["result_dir"] = os.path.join(config["result_dir"], "spec_snr")
        method = "mean" if "mean" in mask_type else "median"
        cut = int(mask_type.split("cut")[-1])
        if cut == 0:
            id_sel = None
        else:
            id_sel = get_id_masks(data_type, "spec_snr", method=method, cut=cut)
    elif "R_cut" in mask_type:
        config["result_dir"] = os.path.join(config["result_dir"], "R_cut")
        cut = float(mask_type.split("R_cut_")[-1])
        id_sel = get_id_masks(data_type, "R_cut", cut=cut)
    elif "mag" in mask_type:
        config["result_dir"] = os.path.join(config["result_dir"], "mag_cut")
        band = mask_type.split("_")[1]
        cut = mask_type.split("_")[3]
        id_sel = get_id_masks(data_type, "mag_cut", band=band, cut=cut)
    elif "zq_cut" in mask_type:
        config["result_dir"] = os.path.join(config["result_dir"], "zq_cut")
        cut = int(mask_type.split("cut")[-1])
        if cut == 0:
            id_sel = None
        else:
            id_sel = get_id_masks(data_type, "zq_cut", cut=cut)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    return id_sel