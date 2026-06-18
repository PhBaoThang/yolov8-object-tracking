# scripts/01_download_dataset.py
"""
Bước 1: Download dataset từ Roboflow và split train/valid (80/20).
"""

import os, shutil, random, yaml

DATASET_URL  = "https://app.roboflow.com/ds/Dj6hml4tOJ?key=DwjIlXyoFu"
ZIP_FILE     = "roboflow_dataset.zip"
DATASET_DIR  = "roboflow_dataset"
RANDOM_SEED  = 42
VALID_RATIO  = 0.2

# ── 1. Download ───────────────────────────────────────────────────────
print(f"📥 Downloading dataset...")
os.system(f'wget -q -O {ZIP_FILE} "{DATASET_URL}"')

# ── 2. Extract ────────────────────────────────────────────────────────
os.makedirs(DATASET_DIR, exist_ok=True)
os.system(f'unzip -q -o {ZIP_FILE} -d {DATASET_DIR}')
print(f"✅ Extracted → {DATASET_DIR}/")

# ── 3. Đọc data.yaml gốc ─────────────────────────────────────────────
yaml_path = f'{DATASET_DIR}/data.yaml'
with open(yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
print(f"Classes: {data_yaml.get('names')}")

# ── 4. Split train → train (80%) / valid (20%) ────────────────────────
train_img_dir = f'{DATASET_DIR}/train/images'
train_lbl_dir = f'{DATASET_DIR}/train/labels'
valid_img_dir = f'{DATASET_DIR}/valid/images'
valid_lbl_dir = f'{DATASET_DIR}/valid/labels'

# Kiểm tra valid đã tồn tại chưa
if os.path.exists(valid_img_dir) and len(os.listdir(valid_img_dir)) > 0:
    print("⚠️  Valid split đã tồn tại — bỏ qua bước split.")
else:
    random.seed(RANDOM_SEED)
    all_images = sorted([
        f for f in os.listdir(train_img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    random.shuffle(all_images)
    n_valid = max(1, int(len(all_images) * VALID_RATIO))
    valid_files = all_images[:n_valid]
    print(f"  Train: {len(all_images)-n_valid} | Valid: {n_valid}")

    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_lbl_dir, exist_ok=True)

    for img_file in valid_files:
        stem = os.path.splitext(img_file)[0]
        shutil.copy(f'{train_img_dir}/{img_file}', f'{valid_img_dir}/{img_file}')
        lbl = f'{train_lbl_dir}/{stem}.txt'
        if os.path.exists(lbl):
            shutil.copy(lbl, f'{valid_lbl_dir}/{stem}.txt')
        os.remove(f'{train_img_dir}/{img_file}')
        if os.path.exists(lbl):
            os.remove(lbl)
    print("✅ Split xong!")

# ── 5. Cập nhật data.yaml ─────────────────────────────────────────────
fixed_yaml = {
    'path' : os.path.abspath(DATASET_DIR),
    'train': 'train/images',
    'val'  : 'valid/images',
    'nc'   : data_yaml.get('nc', 1),
    'names': data_yaml.get('names', ['object']),
}
with open(yaml_path, 'w') as f:
    yaml.dump(fixed_yaml, f, default_flow_style=False, allow_unicode=True)
print("✅ data.yaml đã cập nhật.")

# ── 6. Verify ─────────────────────────────────────────────────────────
for split in ['train', 'valid']:
    n = len(os.listdir(f'{DATASET_DIR}/{split}/images'))
    print(f"  {split}/images: {n} files")