
import os
import sys 
import inspect
import numpy as np
import torch
import torch.nn as nn

class SpecDataLoader:
    def __init__(self, name):
        self.name = name
        self.dataset = name

    def _get_colnames(self, data):
        if hasattr(data, "colnames"):
            return list(data.colnames)
        elif hasattr(data, "keys"):
            return list(data.keys())
        else:
            raise TypeError("Input data must support .colnames or .keys()")

    def _get_meta(self, data, fields=("TARGETID", "RA", "DEC", "ZERR")):
        meta = {}
        colnames = self._get_colnames(data)
        for f in fields:
            if f in colnames:
                meta[f.lower()] = data[f]
        return meta

    def load_data(self, name=None, columns=None, memmap=False):
        from astropy.table import Table
        from helper import READY_FILE
        if name is None: name = self.name
        if name not in READY_FILE:
            raise ValueError(f"Dataset '{name}' not ready")
        fn = READY_FILE[name]
        data = Table.read(fn, memmap=memmap)
        if columns is not None:
            missing = [c for c in columns if c not in data.colnames]
            if missing: raise KeyError(f"Missing columns: {missing}")
            data = data[columns]
        return data

    def chunk_data(self, batch_size=32, max_samples=None, data=None,):
        batches = []
        if data is None: data = self.load_data()
        total_samples = len(data) if max_samples is None else min(max_samples, len(data))
        num_batches = (total_samples + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            batch_data = data[start_idx:end_idx]
            batches.append(batch_data)
        return batches

class SpecFeatureExtractor:
    def __init__(self, device='cuda', model_path = r"/home/pub/jiaqi/OneAstronomy_model/aion-base"):
        sys.path.insert(0, "/home/pub/jiaqi/OneAstronomy_model")
        from aion.model import AION
        from aion.codecs import CodecManager
        # off-line mode to avoid uploading model from hugging-face 
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        self.device = device
        self.model = AION.from_pretrained(model_path, local_files_only=True).to(device).eval()
        self.codec_manager = CodecManager(device=device)

    def _to_tensor(self, data_array, dtype="float32"):
        if dtype == "bool":
            np_dtype = np.bool_
        elif dtype == "float32":
            np_dtype = np.float32
        elif dtype == "int":
            np_dtype = np.int32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        arr = np.array(data_array, dtype=np_dtype, copy=True)
        return torch.tensor(arr, device=self.device)

    def build_modalities(self, data, kind=("desi_spectrum", "legacy_image", "legacy_photometry")):
        from aion.modalities import (
            LegacySurveyImage,
            LegacySurveyFluxG,
            LegacySurveyFluxR,
            LegacySurveyFluxI,
            LegacySurveyFluxZ,
            DESISpectrum,
        )
        kind = set(kind)
        modalities = {}
        if "legacy_image" in kind:
            modalities["image"] = LegacySurveyImage(
                flux=self._to_tensor(data["legacysurvey_image_flux"]),
                bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
                )
        if "legacy_photometry" in kind:
            modalities["photometry"] = {
                "g": LegacySurveyFluxG(value=self._to_tensor(data["legacysurvey_FLUX_G"])),
                "r": LegacySurveyFluxR(value=self._to_tensor(data["legacysurvey_FLUX_R"])),
                "i": LegacySurveyFluxI(value=self._to_tensor(data["legacysurvey_FLUX_I"])),
                "z": LegacySurveyFluxZ(value=self._to_tensor(data["legacysurvey_FLUX_Z"])),
                }
        if "desi_spectrum" in kind:
            modalities["spectrum"] = DESISpectrum(
                flux=self._to_tensor(data["desi_spectrum_flux"]),
                ivar=self._to_tensor(data["desi_spectrum_ivar"]),
                mask=self._to_tensor(data["desi_spectrum_mask"], dtype="bool"),
                wavelength=self._to_tensor(data["desi_spectrum_lambda"]),
                )
        return modalities

    def get_labels(self, data, name = "provabgs", qu=None, as_tensor=False):
        GROUND_TRUTH_MAP = {
        "desi-sv1": {"z": "Z_HP"},
        "provabgs": {"z": "Z_HP", "m_star":"LOG_MSTAR", "z_mw": "LOG_Z_MW", "t_age": "TAGE_MW", "sfr": "sSFR"},
        }
        if name not in GROUND_TRUTH_MAP:
           raise ValueError(f"No ground truth map for dataset '{name}'")
        mapping = GROUND_TRUTH_MAP[name]
        qs = {}
        if qu is None: 
            qu = mapping.keys()
        for q in qu:
            if q not in mapping: raise KeyError(f"Unknown label '{q}'. Available: {list(mapping.keys())}")
            col = mapping[q]
            value = data[col]
            if as_tensor: value = self._to_tensor(value)
            qs[q] = value
        return qs

    @torch.no_grad()
    def extract_features(self, modalities, flatten=True):
        """
        modalities: dict returned by build_modalities(...)
        """
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
            raise ValueError("No modalities were built. Check 'kind'.")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = self.codec_manager.encode(*modality_list)
            # Replace this with the correct AION feature API
            features = self.model.encode(tokens)
        features = features.detach().float().cpu()
        if flatten and features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        return features

