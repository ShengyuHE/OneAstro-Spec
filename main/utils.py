
import os
import sys 
import inspect
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
    
from helper import setup_logging, _decode_array
setup_logging()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Spec_utils') 

def _cast_labels(labels, label_dtype="auto"):
    if labels is None:
        return None
    if label_dtype == "float":
        return labels.astype(np.float32)
    elif label_dtype == "object":
        return labels.astype(object)
    elif label_dtype == "auto":
        return labels
    else:
        raise ValueError(f"Unsupported label_dtype: {label_dtype}")

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
        ## TODO: add reading .h5 file
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
    def __init__(self, device='cuda', model_root=  "/home/pub/jiaqi/OneAstronomy_model/aion-base"):
        sys.path.insert(0, model_root)
        from aion.model import AION
        from aion.codecs import CodecManager
        # off-line mode to avoid uploading model from hugging-face 
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        self.device = device
        self.model = AION.from_pretrained(model_root, local_files_only=True).to(device).eval()
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

    def get_labels(self, data, name="provabgs-v2", qu=None, as_tensor=False):
        from helper import GROUND_LABEL_MAP
        if name not in GROUND_LABEL_MAP:
            raise ValueError(f"No ground truth map for dataset '{name}'")
        mapping = GROUND_LABEL_MAP[name]
        qs = {}
        if qu is None:
            qu = mapping.keys()
        for q in qu:
            if q not in mapping: raise KeyError(f"Unknown label '{q}'. Available: {list(mapping.keys())}")
            col = mapping[q]
            value = np.asarray(data[col])
            # decode byte strings to normal Python strings
            value = _decode_array(value)
            if as_tensor:
                value = self._to_tensor(value)
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
    
class SpecFeatureLoader:
    def __init__(self, dataset="provabgs-v2", loader=None, extractor=None):
        self.dataset = dataset
        self.loader = loader or SpecDataLoader(dataset)
        self.extractor = extractor or SpecFeatureExtractor()  

    def _make_id_mask(self, ids, id_sel=None):
        if id_sel is None:
            return np.ones(len(ids), dtype=bool)
        ids = np.asarray(ids).astype(str)
        id_sel = _decode_array(id_sel)
        id_set = set(id_sel.tolist())
        return np.array([x in id_set for x in ids], dtype=bool)
    
    def _select_labels(self, ids, all_labels, all_label_names, label_names, label_dtype="auto"):
        if label_names is None:
            return None
        cols = []
        for name in label_names:
            if name == "id":
                cols.append(np.asarray(ids, dtype=object))
            else:
                if name not in all_label_names:
                    raise KeyError(f"Requested label '{name}' not found in labels: {all_label_names}")
                idx = all_label_names.index(name)
                cols.append(all_labels[:, idx])
        labels = np.column_stack(cols)
        return _cast_labels(labels, label_dtype=label_dtype)

    def load_features(self, kind=("desi_spectrum",), 
                    label_names=("z",), label_dtype="auto",
                    batch_size=30, max_samples=21870,
                    feature_fn=None, id_sel=None,
                    overwrite=False,):
        if (feature_fn is not None) and (not os.path.exists(feature_fn)):
            logger.warning(f"{feature_fn} does not exist; extracting features from dataset")
        if feature_fn is not None and os.path.exists(feature_fn) and not overwrite:
            logger.info(f"Loading cached features: {feature_fn}")
            cache = np.load(feature_fn, allow_pickle=True)
            features = cache["features"]
            ids = _decode_array(cache["ids"]).astype(str)
            all_labels = cache["labels"]
            all_label_names = list(cache["label_names"])
            mask_ids = self._make_id_mask(ids, id_sel=id_sel)
            ids = ids[mask_ids]
            features = features[mask_ids]
            all_labels = all_labels[mask_ids]
            labels = self._select_labels(ids, all_labels, all_label_names, label_names, label_dtype)
            return features, labels
        logger.info(f"Extract features for {self.dataset} {kind}")
        data = self.loader.load_data()
        if id_sel is not None:
            data_ids = _decode_array(data["TARGETID"]).astype(str)
            mask_ids = self._make_id_mask(data_ids, id_sel=id_sel)
            data = data[mask_ids]
        if max_samples is not None:
            data = data[:max_samples]
        batches = self.loader.chunk_data(batch_size=batch_size, data=data)
        all_features = []
        all_targetid = []
        all_labels = []
        all_label_names = None
        for i, batch in enumerate(tqdm(batches, desc="Feature Extraction", dynamic_ncols=True)):
            # logger.info(modalities)
            #  Features
            modalities = self.extractor.build_modalities(batch, kind=kind)
            features = self.extractor.extract_features(modalities, flatten=False)
            all_features.append(features.numpy().astype(np.float32))
            # TARGETID
            batch_targetid = _decode_array(batch["TARGETID"]).astype(str)
            all_targetid.append(batch_targetid)
            # labels
            labels_dict = self.extractor.get_labels(batch, name=self.dataset, qu=None, as_tensor=False)
            if all_label_names is None:
                all_label_names = list(labels_dict.keys())
            label_array = np.column_stack([np.asarray(labels_dict[name], dtype=object) for name in all_label_names])
            all_labels.append(label_array)
            if i % 10 == 0:
                torch.cuda.empty_cache()
        all_targetid = np.concatenate(all_targetid)
        all_features = np.vstack(all_features)
        all_labels = np.vstack(all_labels)
        if feature_fn is not None:
            os.makedirs(os.path.dirname(feature_fn), exist_ok=True)
            np.savez_compressed(feature_fn, 
                                ids=all_targetid,
                                features=all_features, 
                                labels=all_labels, 
                                label_names=np.array(all_label_names, dtype=object))
            logger.info(f"Save extracted features in: {feature_fn}")
        labels = self._select_labels(all_targetid, all_labels, all_label_names, label_names, label_dtype)
        return all_features, labels

    def update_feature_labels(self, feature_fn, label_names=("z",), saving=False):
        """
        Update labels inside an existing cached feature file without recomputing features.
        """
        from helper import GROUND_LABEL_MAP
        if not os.path.exists(feature_fn):
            raise FileNotFoundError(f"Feature file does not exist: {feature_fn}")
        logger.info(f"Loading cached features: {feature_fn}")
        cache = np.load(feature_fn, allow_pickle=True)
        cache_features = cache["features"]
        cache_label_names = cache["label_names"]
        cache_ids = _decode_array(cache["ids"]).astype(str)
        logger.info(f"Update label: {cache_label_names} -> {label_names}")
        # Reload source table
        data = self.loader.load_data()
        data_ids = _decode_array(data["TARGETID"]).astype(str)
        id_to_index = {tid: i for i, tid in enumerate(data_ids)}
        logger.info(f"Dataset size: {len(data_ids)}")
        logger.info(f"Cached feature size: {len(cache_ids)}")
        if self.dataset not in GROUND_LABEL_MAP:
            raise ValueError(f"No ground truth map for dataset '{self.dataset}'")
        mapping = GROUND_LABEL_MAP[self.dataset]
        # Build new label matrix
        new_labels = []
        for lab in label_names:
            if lab not in mapping:
                raise KeyError(f"Unknown label '{lab}'. Available labels: {list(mapping.keys())}")
            colname = mapping[lab]
            if colname not in data.colnames:
                raise KeyError(f"Mapped column '{colname}' for label '{lab}' not found in dataset")
            col = np.asarray(data[colname])
            matched = []
            for tid in cache_ids:
                if tid not in id_to_index:
                    raise KeyError(f"TARGETID {tid} not found in dataset")
                matched.append(col[id_to_index[tid]])
            matched = _decode_array(np.asarray(matched))
            new_labels.append(matched)
        new_labels = np.column_stack(new_labels)
        if len(cache_features) != len(new_labels):
            raise ValueError(f"Feature/label size mismatch: features={len(cache_features)}, labels={len(new_labels)}")
        if saving:
            np.savez_compressed(feature_fn,
                                ids=cache_ids,
                                features=cache_features, 
                                labels=new_labels,
                                label_names=np.array(label_names, dtype=object),)
        return cache_features, new_labels
    
    def extend_feature_cache(self, feature_fn, new_dataset=None, kind=("desi_spectrum",), saving=False):
        ## TODO: add function to extend the features with extra dataset
        if feature_fn is None:
            raise ValueError("feature_fn must be provided")
        if not os.path.exists(feature_fn):
            raise FileNotFoundError(f"Feature file does not exist: {feature_fn}")


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