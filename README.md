# OneAstro-Spec

**OneAstro-Spec** is a spectral foundation model for astronomy, fine‑tuned from the multimodal model (currently AION-1, future OneAstronomy). It aims to support multiple spectroscopic tasks across surveys like DESI, 4MOST, Euclid and so on .

## Key Tasks
- **Redshift estimation** (tested on DESI DR1 and 4MOST)
- **Galaxy property inference** (velocity dispersion, SFR, stellar mass, age, metallicity, AGN)
- **Object classification** (star/galaxy/QSO, fine classes: MW/BGS/LRG/ELG/QSO)
- **Retrieval** (similar spectra, morphology matching)
- **Generative** (image from spectrum) and **cross‑modal matching**

## Model
Derived from **AION-1** (multimodal transformer trained on images, spectra, photometry, text), later  **OneAstronomy‑v1**. Adapted via post‑training on spectroscopic data.

## Data
Public DESI DR1 spectra and value‑added catalogs (redshifts, stellar masses, emission lines) + Legacy Survey images, Galaxy Zoo 10, PROVABGS, and simulated spectra.  
*Internal mirror at Zhejiang Lab – see [data notes](docs/DATA_ACCESS.md).*

## Quick Start
```bash
conda activate SpecFun

