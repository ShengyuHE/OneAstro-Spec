import os
import re
import h5py
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm

CATALOG_ROOT = "/mnt/oss_nanhu100TB/default/100TB_scientificData/desi_provabgs_hdf5_data"
SPECTRUM_ROOT = "/mnt/oss_nanhu100TB/default/100TB_scientificData/desi_hdf5_data"
IMAGE_PARENT = "/mnt/oss_nanhu100TB/default/zjq/results/ls_desi_provabgs"
OUTPUT_ROOT = "/mnt/oss_nanhu100TB/default/zjq/results/SpecFun/features/provabgs-full"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def extract_hpix(filename):
    match = re.search(r'healpix[=_](\d+)', filename)
    return match.group(1) if match else None

def build_image_path_map(root_dir):
    path_map = {}
    print(f"开始扫描图像目录: {root_dir}")
    try:
        workers = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except Exception as e:
        print(f"无法读取目录 {root_dir}: {e}")
        return path_map

    for worker in tqdm(workers, desc="构建图像索引"):
        worker_path = os.path.join(root_dir, worker)
        try:
            hpix_dirs = os.listdir(worker_path)
            for h_dir in hpix_dirs:
                if "healpix" in h_dir:
                    hpix_num = re.search(r'(\d+)', h_dir).group(1)
                    target_path = os.path.join(worker_path, h_dir)
                    files = [f for f in os.listdir(target_path) if f.endswith('.hdf5')]
                    if files:
                        path_map[hpix_num] = os.path.join(target_path, files[0])
        except:
            continue
            
    print(f"索引构建完成，共计 {len(path_map)} 个天区的图像已就绪。")
    return path_map

def get_z_value(cf, i):
    """获取红移值，兼容不同字段名"""
    if 'Z' in cf:
        return float(cf['Z'][i])
    # elif 'Z_MW' in cf:
    #     return float(cf['Z_MW'][i])
    elif 'Z_HP' in cf:
        return float(cf['Z_HP'][i])
    else:
        return 0.0

def get_logmstar_value(cf, i):
    """获取恒星质量对数，取中位数"""
    if 'PROVABGS_LOGMSTAR' in cf:
        data = cf['PROVABGS_LOGMSTAR'][i]
        return float(np.median(data))
    elif 'PROVABGS_LOGMSTAR_BF' in cf:
        return float(cf['PROVABGS_LOGMSTAR_BF'][i])
    else:
        return -1.0

def process_single_healpix_streaming(hpix, img_path):
    """
    流式处理：匹配一条就保存一条到 HDF5 文件
    """
    cat_path = os.path.join(CATALOG_ROOT, f"healpix_{hpix}_001-of-001.h5")
    spec_path = os.path.join(SPECTRUM_ROOT, f"healpix_{hpix}_001-of-001.hdf5")
    out_path = os.path.join(OUTPUT_ROOT, f"ready_hpix_{hpix}.h5")
    
    if not os.path.exists(cat_path) or not os.path.exists(spec_path):
        return 0, "MISSING_DATA_FILES"

    tqdm.write(f"🔍 [HPIX {hpix}] 开始处理...")
    saved_count = 0
    
    with h5py.File(cat_path, 'r') as cf, \
         h5py.File(spec_path, 'r') as sf, \
         h5py.File(img_path, 'r') as imf:
        
        cat_ids = cf['object_id'][:].astype(str)
        cat_ras = cf['ra'][:]
        cat_decs = cf['dec'][:]
        cat_len = len(cat_ids)
        tqdm.write(f"  Catalog: {cat_len} 条")
        
        spec_ids = sf['TARGETID'][:].astype(str)
        spec_id_map = {tid: i for i, tid in enumerate(spec_ids)}
        tqdm.write(f"  Spectrum: {len(spec_ids)} 条")
        
        img_ras = imf['ra'][:]
        img_decs = imf['dec'][:]
        tqdm.write(f"  Image: {len(img_ras)} 条")
        img_catalog = SkyCoord(ra=img_ras * u.degree, dec=img_decs * u.degree)
        
        cat_coords = SkyCoord(ra=cat_ras * u.degree, dec=cat_decs * u.degree)
        tqdm.write(f"  执行空间匹配...")
        idx_img, d2d_img, _ = cat_coords.match_to_catalog_sky(img_catalog)
        max_sep = 1.0 * u.arcsec
        
        valid_idx = [i for i, tid in enumerate(cat_ids) if tid in spec_id_map and d2d_img[i] < max_sep]
        total_valid = len(valid_idx)
        tqdm.write(f"  匹配成功 {total_valid} 条，开始流式保存...")
        
        if not valid_idx:
            return 0, "NO_MATCH_FOUND"

        with h5py.File(out_path, 'w') as outf:
            sample_i = valid_idx[0]
            sample_s_idx = spec_id_map[cat_ids[sample_i]]
            sample_i_idx = idx_img[sample_i]
            sample_flux_shape = sf['spectrum_flux'][sample_s_idx].shape
            sample_img_shape = imf['image_array'][sample_i_idx].shape
            sample_lambda_shape = sf['spectrum_lambda'][sample_s_idx].shape
            
            max_len = len(valid_idx)
            outf.create_dataset('TARGETID', (max_len,), dtype=h5py.special_dtype(vlen=str))
            outf.create_dataset('RA', (max_len,), dtype=np.float64)
            outf.create_dataset('DEC', (max_len,), dtype=np.float64)
            outf.create_dataset('Z', (max_len,), dtype=np.float64)
            outf.create_dataset('PROVABGS_LOGMSTAR', (max_len,), dtype=np.float64)
            outf.create_dataset('desi_spectrum_flux', (max_len,) + sample_flux_shape, dtype=np.float32)
            outf.create_dataset('desi_spectrum_ivar', (max_len,) + sample_flux_shape, dtype=np.float32)
            outf.create_dataset('desi_spectrum_mask', (max_len,) + sample_flux_shape, dtype=np.bool_)
            outf.create_dataset('desi_spectrum_lambda', (max_len,) + sample_lambda_shape, dtype=np.float32)
            outf.create_dataset('legacysurvey_image_flux', (max_len,) + sample_img_shape, dtype=np.float32)
            outf.create_dataset('legacysurvey_FLUX_G', (max_len,), dtype=np.float32)
            outf.create_dataset('legacysurvey_FLUX_R', (max_len,), dtype=np.float32)
            outf.create_dataset('legacysurvey_FLUX_I', (max_len,), dtype=np.float32)
            outf.create_dataset('legacysurvey_FLUX_Z', (max_len,), dtype=np.float32)
            
            for out_idx, i in enumerate(tqdm(valid_idx, desc=f"  保存进度", leave=False)):
                tid = cat_ids[i]
                s_idx = spec_id_map[tid]
                i_idx = idx_img[i]
                
                outf['TARGETID'][out_idx] = tid
                outf['RA'][out_idx] = cat_ras[i]
                outf['DEC'][out_idx] = cat_decs[i]
                outf['Z'][out_idx] = get_z_value(cf, i)
                outf['PROVABGS_LOGMSTAR'][out_idx] = get_logmstar_value(cf, i)
                
                outf['desi_spectrum_flux'][out_idx] = sf['spectrum_flux'][s_idx].astype(np.float32)
                outf['desi_spectrum_ivar'][out_idx] = sf['spectrum_ivar'][s_idx].astype(np.float32)
                outf['desi_spectrum_mask'][out_idx] = sf['spectrum_mask'][s_idx].astype(bool)
                outf['desi_spectrum_lambda'][out_idx] = sf['spectrum_lambda'][s_idx].astype(np.float32)
                
                outf['legacysurvey_image_flux'][out_idx] = imf['image_array'][i_idx].astype(np.float32)
                outf['legacysurvey_FLUX_G'][out_idx] = imf['FLUX_G'][i_idx].astype(np.float32)
                outf['legacysurvey_FLUX_R'][out_idx] = imf['FLUX_R'][i_idx].astype(np.float32)
                outf['legacysurvey_FLUX_I'][out_idx] = imf['FLUX_I'][i_idx].astype(np.float32)
                outf['legacysurvey_FLUX_Z'][out_idx] = imf['FLUX_Z'][i_idx].astype(np.float32)
                
                saved_count += 1
                
                if saved_count % 50 == 0:
                    tqdm.write(f"    已保存 {saved_count}/{total_valid} 条")

    return saved_count, "SUCCESS"

if __name__ == "__main__":
    img_map = build_image_path_map(IMAGE_PARENT)
    cat_files = sorted([f for f in os.listdir(CATALOG_ROOT) if f.startswith("healpix_") and f.endswith(".h5")])
    print(f"待处理天区总数: {len(cat_files)}")

    total_saved = 0
    for f_name in tqdm(cat_files, desc="总进度"):
        hpix = extract_hpix(f_name)
        if hpix in img_map:
            try:
                count, status = process_single_healpix_streaming(hpix, img_map[hpix])
                if status == "SUCCESS":
                    total_saved += count
                    tqdm.write(f"✅ HPIX {hpix}: 保存 {count} 条 (累计: {total_saved})")
                else:
                    tqdm.write(f"⚠️ HPIX {hpix}: {status}")
            except Exception as e:
                tqdm.write(f"❌ HPIX {hpix} 处理异常: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n所有天区处理完成，共保存 {total_saved} 条记录。")
