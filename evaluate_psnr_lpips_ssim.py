import os
import torch
import lpips
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr

GT_BASE = "ActiveNeRF-GPU4EDU/Original-Test-GPUEDU Complete/ActiveNeRF/data/nerf_llff_data"
PRED_BASE = "ActiveNeRF-GPU4EDU/Original-Test-GPUEDU Complete/ActiveNeRF/logs"
RESOLUTION = (756, 1008)
ITERATIONS = ["050000", "100000", "150000", "200000"]
HOLD = 8  # Corresponds to llffhold=8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='vgg').to(device)

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return -10 * np.log10(mse)

def load_image(path, size):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)

def evaluate_folder(scene):
    gt_dir = os.path.join(GT_BASE, scene)
    # added
    pred_dir = os.path.join(PRED_BASE, f"llff_{scene}")
    # pred_dir = os.path.join(PRED_BASE, f"plugandplay_llff_{scene}") # LOG
    # pred_dir = os.path.join(PRED_BASE, f"plugandplayLogHighScale_llff_{scene}") # LOG HighScale
    # pred_dir = os.path.join(PRED_BASE, f"plugandplayLinear_llff_{scene}") # Linear
    # pred_dir = os.path.join(PRED_BASE, f"plugandplayLinearHighScale_llff_{scene}") # Linear HighScale
    
    # pred_dir = os.path.join(PRED_BASE, f"Special_llff_{scene}") # Special Loss
    # pred_dir = os.path.join(PRED_BASE, f"Special_1mean-KL_llff_{scene}") # Special 1mean KL Loss
    # pred_dir = os.path.join(PRED_BASE, f"Special-1mean-noKL_llff_{scene}") # Special 1mean noKL Loss
    # pred_dir = os.path.join(PRED_BASE, f"SpecialHighWeight_llff_{scene}") # Special Weighted Loss
    # pred_dir = os.path.join(PRED_BASE, f"SpecialHighWeight-1mean-KL_llff_{scene}") # Special Weighted 1mean KL Loss
    # pred_dir = os.path.join(PRED_BASE, f"SpecialHighWeight-1mean-noKL_llff_{scene}") # Special Weighted 1mean noKL Loss

    if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
        print(f"Skipping scene: {scene} (directory missing)")
        return None, None

    gt_img_dir = sorted([os.path.join(gt_dir, d) for d in os.listdir(gt_dir)
                         if d.startswith("images") and os.path.isdir(os.path.join(gt_dir, d))])[0]
    gt_files = sorted([os.path.join(gt_img_dir, f) for f in os.listdir(gt_img_dir)
                        if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".JPG")])[::HOLD]

    lpips_vals, ssim_vals, psnr_vals = [], [], []

    for it in ITERATIONS:
        pred_it_dir = os.path.join(pred_dir, f"testset_{it}")
        if not os.path.isdir(pred_it_dir):
            lpips_vals.append(np.nan)
            ssim_vals.append(np.nan)
            psnr_vals.append(np.nan)
            continue

        pred_files = sorted([os.path.join(pred_it_dir, f) for f in os.listdir(pred_it_dir)
                             if f.endswith(".png") and "_uncert" not in f])

        gt_imgs = [load_image(p, RESOLUTION).to(device) for p in gt_files]
        pred_imgs = [load_image(p, RESOLUTION).to(device) for p in pred_files]

        if len(gt_imgs) == 0 or len(pred_imgs) == 0:
            print(f"Warning: No images found in scene {scene}, iteration {it}. Skipping.")
            lpips_vals.append(np.nan)
            ssim_vals.append(np.nan)
            psnr_vals.append(np.nan)
            continue
        
        if len(gt_imgs) != len(pred_imgs):
            print(f"Warning: Ground truth and prediction count mismatch in scene {scene}, iteration {it}. Skipping.")
            lpips_vals.append(np.nan)
            ssim_vals.append(np.nan)
            psnr_vals.append(np.nan)
            continue
        
        lpips_scores, ssim_scores, psnr_scores = [], [], []
        for gt, pred in zip(gt_imgs, pred_imgs):
            with torch.no_grad():
                lpips_val = lpips_fn(gt, pred).item()
            gt_np = gt.squeeze(0).cpu().permute(1, 2, 0).numpy()
            pred_np = pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
            ssim_val = ssim(gt_np, pred_np, channel_axis=2, data_range=1.0)
            # ssim_val = ssim(gt_np, pred_np, channel_axis=2, data_range=225.0)
            psnr_val = compute_psnr(gt_np, pred_np)
            # psnr_val = psnr(gt_np, pred_np, data_range=1.0)

            lpips_scores.append(lpips_val)
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)

        lpips_vals.append(np.mean(lpips_scores))
        ssim_vals.append(np.mean(ssim_scores))
        psnr_vals.append(np.mean(psnr_scores))


    lpips_vals.append(np.mean([v for v in lpips_vals if not np.isnan(v)]))
    ssim_vals.append(np.mean([v for v in ssim_vals if not np.isnan(v)]))
    psnr_vals.append(np.mean([v for v in psnr_vals if not np.isnan(v)]))
    
    return lpips_vals, ssim_vals, psnr_vals

if __name__ == '__main__':
    # skip_scenes = [] # added
    skip_scenes = ['horns', 'leaves', 'orchids', 'room', 'trex'] # added
    # skip_scenes = ['horns', 'orchids', 'room', 'trex'] # added
    scenes = sorted([d for d in os.listdir(GT_BASE) if os.path.isdir(os.path.join(GT_BASE, d))])
    lpips_result, ssim_result, psnr_result = {}, {}, {}

    # added
    for scene in scenes:
        if scene in skip_scenes:
            print(f"Skipping scene: {scene} (manually excluded)")
            continue
        
        print(f"Evaluating scene: {scene}")
        lpips_vals, ssim_vals, psnr_vals = evaluate_folder(scene)
        if lpips_vals is not None:
            key = scene.capitalize()
            lpips_result[key] = lpips_vals
            ssim_result[key] = ssim_vals
            psnr_result[key] = psnr_vals

    rows = ["50k", "100k", "150k", "200k", "average"]
    df_lpips = pd.DataFrame(lpips_result, index=rows)
    df_ssim = pd.DataFrame(ssim_result, index=rows)
    df_psnr = pd.DataFrame(psnr_result, index=rows)
    
    df_ssim["averages"] = df_ssim.mean(axis=1)
    df_lpips["averages"] = df_lpips.mean(axis=1)
    df_psnr["averages"] = df_psnr.mean(axis=1)

    # with open("LPIPS_SSIM_PSNR_RESULTS_ActiveNeRF-ALL.txt", "w") as f:
    with open("LPIPS_SSIM_PSNR_RESULTS_ActiveNeRF.txt", "w") as f:
    # with open("LPIPS_SSIM_PSNR_RESULTS_LOG.txt", "w") as f:
    # with open("LPIPS_SSIM_PSNR_RESULTS_LOGHighScale.txt", "w") as f:
    # with open("LPIPS_SSIM_PSNR_RESULTS_Linear.txt", "w") as f:
    # with open("LPIPS_SSIM_PSNR_RESULTS_LinearHighScale.txt", "w") as f:    
    # with open("LPIPS_SSIM_PSNR_RESULTS_Special.txt", "w") as f:
    # with open("LPIPS_SSIM_PSNR_RESULTS_Special-1mean-KL.txt", "w") as f:
    # with open("LPIPS_SSIM_PSNR_RESULTS_Special-1mean-noKL.txt", "w") as f:
    # with open("LPIPS_SSIM_PSNR_RESULTS_SpecialHighWeight.txt", "w") as f:
    # with open("LPIPS_SSIM_PSNR_RESULTS_SpecialHighWeight-1mean-KL.txt", "w") as f:
    # with open("LPIPS_SSIM_PSNR_RESULTS_SpecialHighWeight-1mean-noKL.txt", "w") as f:
        f.write("SSIM\n")
        f.write(df_ssim.to_string())
        f.write("\n\nLPIPS\n")
        f.write(df_lpips.to_string())
        f.write("\n\nPSNR\n")
        f.write(df_psnr.to_string())

    print("Results saved to LPIPS_SSIM_PSNR_RESULTS.txt")
