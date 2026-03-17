import os, sys 
import logging 
import argparse
import numpy as np
from astropy.table import Table

sys.path.append("..")
from utils import SpecDataLoader
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