import nibabel as nib
import numpy as np
import os
import shutil
import random
from pathlib import Path
import cv2
from tqdm import tqdm

VOLUME_SLICES = 60
VOLUME_START_AT = 10
IMG_SIZE = 640
BLACK_THRESHOLD = 0.8 # ELIMINAR LAS IMAGENES EN NEGRO

rawdata_dir = "03_DataSet/rawdata"
derivatives_dir = "03_DataSet/derivatives"
output_img_dir = "imagenes"
output_lbl_dir = "labels"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

samples = []
sample_descarte = []

def es_mayoritariamente_negra(imagen, threshold=BLACK_THRESHOLD):
    total_pixels = imagen.size
    pixeles_negros = np.sum(imagen == 0)
    proporcion = pixeles_negros / total_pixels
    return proporcion >= threshold

def import_data():
    for archivo in tqdm(range(1, 251)):
        try:
            path_dwi = f"{rawdata_dir}/sub-strokecase{archivo:04d}/ses-0001/sub-strokecase{archivo:04d}_ses-0001_dwi.nii.gz"
            path_msk = f"{derivatives_dir}/sub-strokecase{archivo:04d}/ses-0001/sub-strokecase{archivo:04d}_ses-0001_msk.nii.gz"

            archivo_dwi = nib.load(path_dwi)
            archivo_msk = nib.load(path_msk)

            imagen_dwi = archivo_dwi.get_fdata()
            imagen_msk = archivo_msk.get_fdata()

            if imagen_dwi.shape[2] > VOLUME_SLICES + VOLUME_START_AT:
                samples.append(f'sub-strokecase{archivo:04d}')
                for j in range(VOLUME_SLICES):
                    slice_img = cv2.resize(imagen_dwi[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                    slice_lbl = cv2.resize(imagen_msk[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

                    slice_img_norm = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    slice_lbl_norm = cv2.normalize(slice_lbl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    if not es_mayoritariamente_negra(slice_img_norm):
                        cv2.imwrite(os.path.join(output_img_dir, f"sub{archivo:04d}_slice{j:02d}.jpg"), slice_img_norm)
                        cv2.imwrite(os.path.join(output_lbl_dir, f"sub{archivo:04d}_slice{j:02d}.jpg"), slice_lbl_norm)
            else:
                sample_descarte.append(f'sub-strokecase{archivo:04d}')
        except FileNotFoundError:
            sample_descarte.append(f'sub-strokecase{archivo:04d}')

def create_txt_yolo():
    label_dir = "labels"
    yolo_dir = os.path.join(label_dir, "yolov11")
    os.makedirs(yolo_dir, exist_ok=True)

    min_area = 600

    def obtener_bbox_yolo(mask):
        kernel = np.ones((5, 5), np.uint8)
        mask_dilatada = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                img_h, img_w = mask.shape

                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width = w / img_w
                height = h / img_h

                bboxes.append((0, x_center, y_center, width, height))

        return bboxes

    for nombre_archivo in os.listdir(label_dir):
        path_imagen = os.path.join(label_dir, nombre_archivo)

        if nombre_archivo.endswith(".jpg") and os.path.isfile(path_imagen):
            mask = cv2.imread(path_imagen, cv2.IMREAD_GRAYSCALE)

            _, mask_binaria = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            resultados = obtener_bbox_yolo(mask_binaria)

            nombre_txt = os.path.splitext(nombre_archivo)[0] + ".txt"
            path_txt = os.path.join(yolo_dir, nombre_txt)

            if resultados:
                with open(path_txt, "w") as f:
                    for resultado in resultados:
                        clase, x_c, y_c, w, h = resultado
                        f.write(f"{clase} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            else:
                open(path_txt, "w").close()

def create_path():
    src_images = Path("imagenes")
    src_labels = Path("labels/yolov11")

    base_dir = Path("data_yolo")
    image_train_dir = base_dir / "images" / "train"
    image_val_dir   = base_dir / "images" / "val"
    label_train_dir = base_dir / "labels" / "train"
    label_val_dir   = base_dir / "labels" / "val"

    for d in [image_train_dir, image_val_dir, label_train_dir, label_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    archivos = [f.stem for f in src_images.glob("*.jpg")]
    random.shuffle(archivos)

    split_idx = int(0.8 * len(archivos))
    train_archivos = archivos[:split_idx]
    val_archivos = archivos[split_idx:]

    def mover_archivos(archivos, image_dst, label_dst):
        for name in archivos:
            img_path = src_images / f"{name}.jpg"
            label_path = src_labels / f"{name}.txt"

            if img_path.exists():
                shutil.copy(img_path, image_dst / f"{name}.jpg")
            if label_path.exists():
                shutil.copy(label_path, label_dst / f"{name}.txt")
            else:
                open(label_dst / f"{name}.txt", 'w').close()

    mover_archivos(train_archivos, image_train_dir, label_train_dir)
    mover_archivos(val_archivos, image_val_dir, label_val_dir)

