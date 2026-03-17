
import os
import sys 
import inspect
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
    
from helper import setup_logging
setup_logging()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Spec_utils') 
    
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
            # LegacySurveyFluxI,
            LegacySurveyFluxZ,
            DESISpectrum,
        )
        kind = set(kind)
        modalities = {}
        if "legacy_image" in kind:
            modalities["image"] = LegacySurveyImage(
                ##TODO: image input bands affect the performance
                flux=self._to_tensor(data["legacysurvey_image_flux"]),
                bands=["DES-G", "DES-R",  "DES-I" ,"DES-Z"],
                )
        if "legacy_photometry" in kind:
            modalities["photometry"] = {
                ##TODO: flux input bands affect the performance
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
        from helper import GROUND_TRUTH_MAP
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
    def extract_features(self, modalities, flatten=False):
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
            # TODO: check the dimension of tokens
            # Replace this with the correct AION feature API
            features = self.model.encode(tokens)
        features = features.detach().float().cpu()
        return features
    
class FeatureDataset(Dataset):
    def __init__(self, features, labels, task="classification"):
        self.features = torch.tensor(features, dtype=torch.float32)
        labels = np.asarray(labels)
        if task == "classification":
            labels = np.squeeze(labels)
            self.labels = torch.tensor(labels, dtype=torch.long)
        elif task == "regression":
            self.labels = torch.tensor(labels, dtype=torch.float32)
            if self.labels.ndim == 1:
                self.labels = self.labels.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown task: {task}")
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def load_features(loader, extractor,
                  dataset="provabgs", kind=("desi_spectrum",), label_names=("z",), label_dtype="auto",
                  batch_size=30, max_samples=21870, save_path=None, overwrite=False):
    # data = loader.load_data(name=dataset)
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        logger.info(f"Loading cached features from {save_path}")
        cache = np.load(save_path, allow_pickle=True)
        features = cache["features"]
        all_labels = cache["labels"]
        all_label_names = list(cache["label_names"])
        missing = [name for name in label_names if name not in all_label_names]
        if missing:
            raise KeyError(f"Requested labels {missing} not found in cached file")
        col_idx = [all_label_names.index(name) for name in label_names]
        labels = all_labels[:, col_idx]
        if label_dtype == "float":
            labels = labels.astype(np.float32)
        elif label_dtype == "object":
            labels = labels.astype(object)
        elif label_dtype == "auto":
            pass
        else:
            raise ValueError(f"Unsupported label_dtype: {label_dtype}")
        return features, labels

    batches = loader.chunk_data(batch_size=batch_size, max_samples=max_samples)
    all_features = []
    all_labels = []
    all_label_names = None
    for i, batch in enumerate(tqdm(batches, desc="Feature Extraction")):
        modalities = extractor.build_modalities(batch, kind=kind)
        features = extractor.extract_features(modalities, flatten=False)
        labels_dict = extractor.get_labels(batch, name=dataset, qu=None, as_tensor=False)
        if all_label_names is None:
            all_label_names = list(labels_dict.keys())
            # print("all_label_names:", all_label_names)
        all_features.append(features.numpy().astype(np.float32))
        label_array = np.column_stack([np.asarray(labels_dict[name], dtype=object) for name in all_label_names])
        all_labels.append(label_array)
        if i % 10 == 0:
            torch.cuda.empty_cache()
    all_features = np.vstack(all_features)
    all_labels = np.vstack(all_labels)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, features=all_features, 
                            labels=all_labels, label_names=np.array(all_label_names, dtype=object))
    col_idx = [all_label_names.index(name) for name in label_names]
    labels = all_labels[:, col_idx]
    if label_dtype == "float":
        labels = labels.astype(np.float32)
    elif label_dtype == "object":
        labels = labels.astype(object)
    elif label_dtype == "auto":
        pass
    else:
        raise ValueError(f"Unsupported label_dtype: {label_dtype}")
    return all_features, labels