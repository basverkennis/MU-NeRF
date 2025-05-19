import os
import cv2
from PIL import Image
import numpy as np

# === CONFIG ===
GT_BASE = "/Users/basverkennis/Desktop/Original-Test-GPUEDU Complete/ActiveNeRF/data/nerf_llff_data"
MODELS = {
    "ActiveNeRF": "/Users/basverkennis/Desktop/Original-Test-GPUEDU Complete/ActiveNeRF/logs",
    "ActiveNeRF-LOG": "/Users/basverkennis/Desktop/Original-Test-GPUEDU Complete/ActiveNeRF/logs"
}
ITERATIONS = ["050000", "100000", "150000", "200000"]
HOLD = 8
ZOOM_BLOCK_SIZE = 80
ZOOM_FACTOR = 3
DISPLAY_SCALE = 0.3

# === HELPERS ===
def load_image_np(path):
    img = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def load_and_resize_gt(path, size):
    img = Image.open(path).convert('RGB')
    img = img.resize(size, Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def get_model_images(scene, model_name, model_base, gt_files):
    pred_img_lists = []

    for it in ITERATIONS:
        if model_name == "ActiveNeRF":
            pred_dir = os.path.join(model_base, f"llff_{scene}", f"testset_{it}")
        elif model_name == "ActiveNeRF-LOG":
            pred_dir = os.path.join(model_base, f"plugandplay_llff_{scene}", f"testset_{it}")
        else:
            print(f"Unknown model name pattern: {model_name}")
            return None

        if not os.path.isdir(pred_dir):
            print(f"Missing prediction directory for {model_name} iter {it} in {scene}")
            return None

        pred_files = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir)
                             if f.endswith(".png") and "_uncert" not in f])

        if len(pred_files) != len(gt_files):
            print(f"Mismatch in GT and PRED count for scene {scene}, model {model_name}, iter {it}")
            return None

        pred_img_lists.append([load_image_np(f) for f in pred_files])

    # Resize GT
    target_shape = pred_img_lists[0][0].shape[:2][::-1]
    gt_imgs = [load_and_resize_gt(f, target_shape) for f in gt_files]

    return list(zip(*pred_img_lists, gt_imgs))  # shape: [views][50k, ..., GT]

def get_all_model_rows(scene):
    gt_dir = os.path.join(GT_BASE, scene)
    gt_img_dir = sorted([os.path.join(gt_dir, d) for d in os.listdir(gt_dir)
                         if d.startswith("images") and os.path.isdir(os.path.join(gt_dir, d))])[0]
    gt_files = sorted([os.path.join(gt_img_dir, f) for f in os.listdir(gt_img_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))])[::HOLD]

    model_data = {}
    for model_name, model_base in MODELS.items():
        model_images = get_model_images(scene, model_name, model_base, gt_files)
        if model_images:
            model_data[model_name] = model_images

    if not model_data:
        return None

    # Merge: per-view, collect [model_A_row, model_B_row, model_C_row]
    n_views = len(next(iter(model_data.values())))
    merged_rows = []
    for view_idx in range(n_views):
        row = []
        for model in MODELS:
            if model in model_data:
                row.append(model_data[model][view_idx])  # shape: [50k, 100k, 150k, 200k, GT]
        merged_rows.append(row)
    return merged_rows

# === GUI ===
def display_gallery(scene, image_rows):
    row_idx = 0
    cursor_pos_scaled = None

    def on_mouse(event, x, y, flags, param):
        nonlocal cursor_pos_scaled
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE:
            cursor_pos_scaled = (x, y)

    cv2.namedWindow("Gallery", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Gallery", on_mouse)

    while True:
        view_rows = image_rows[row_idx]
        combined_rows = []
        for model_images in view_rows:
            row_img = cv2.hconcat(model_images)
            combined_rows.append(row_img)
        combined_full = cv2.vconcat(combined_rows)

        scaled = cv2.resize(combined_full, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)

        if cursor_pos_scaled:
            full_x = int(cursor_pos_scaled[0] / DISPLAY_SCALE)
            full_y = int(cursor_pos_scaled[1] / DISPLAY_SCALE)

            block_half = ZOOM_BLOCK_SIZE // 2
            row_width = view_rows[0][0].shape[1]
            row_height = view_rows[0][0].shape[0]

            zoom_grid = []
            for model_images in view_rows:  # Each model: [50k, 100k, ..., GT]
                zoom_row = []
                for img in model_images:
                    h, w = img.shape[:2]
                    x = np.clip(full_x % row_width, block_half, w - block_half)
                    y = np.clip(full_y % (row_height * len(view_rows)), block_half, h - block_half)
                    patch = img[y - block_half:y + block_half, x - block_half:x + block_half]
                    zoomed_patch = cv2.resize(
                        patch,
                        (ZOOM_BLOCK_SIZE * ZOOM_FACTOR, ZOOM_BLOCK_SIZE * ZOOM_FACTOR),
                        interpolation=cv2.INTER_NEAREST
                    )
                    zoom_row.append(zoomed_patch)
                zoom_grid.append(cv2.hconcat(zoom_row))  # One row per model

            zoom_combined = cv2.vconcat(zoom_grid)
            cv2.imshow("Zoom", zoom_combined)

        cv2.imshow("Gallery", scaled)
        key = cv2.waitKey(30)

        if key == 27: break  # ESC
        elif key == ord('d') or key == 83:
            row_idx = (row_idx + 1) % len(image_rows)
            cursor_pos_scaled = None
        elif key == ord('a') or key == 81:
            row_idx = (row_idx - 1) % len(image_rows)
            cursor_pos_scaled = None

    cv2.destroyAllWindows()

def main():
    scenes = sorted([d for d in os.listdir(GT_BASE) if os.path.isdir(os.path.join(GT_BASE, d))])
    for scene in scenes:
        print(f"Loading {scene}...")
        image_rows = get_all_model_rows(scene)
        if image_rows:
            display_gallery(scene, image_rows)

if __name__ == "__main__":
    main()