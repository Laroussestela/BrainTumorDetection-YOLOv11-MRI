import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage
import cv2

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO


def ground_truth(archivo):
    label = f'Strokecase{archivo:04d}'

    path_adc = f"03_DataSet/rawdata/sub-strokecase{archivo:04d}/ses-0001/sub-strokecase{archivo:04d}_ses-0001_dwi.nii.gz"
    path_msk = f"03_DataSet/derivatives/sub-strokecase{archivo:04d}/ses-0001/sub-strokecase{archivo:04d}_ses-0001_msk.nii.gz"

    archivo_adc = nib.load(path_adc)
    archivo_msk = nib.load(path_msk)
    imagen_adc = archivo_adc.get_fdata()
    imagen_msk = archivo_msk.get_fdata()

    imagen_adc_stack = np.transpose(imagen_adc, (2, 0, 1))
    imagen_msk_stack = np.transpose(imagen_msk, (2, 0, 1))
    imagen_msk_stack = np.clip(imagen_msk_stack, 0, 1)

    img_montage = montage(imagen_adc_stack)
    msk_montage = montage(imagen_msk_stack).astype(np.uint8)

    image_rgb = cv2.cvtColor(img_montage.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    orange = np.zeros_like(image_rgb)
    orange[:, :, 0] = 255  # B
    orange[:, :, 1] = 165  # G
    orange[:, :, 2] = 0    # R

    alpha = 0.5
    mask_bool = msk_montage.astype(bool)
    image_rgb[mask_bool] = cv2.addWeighted(image_rgb[mask_bool], 1 - alpha, orange[mask_bool], alpha, 0)

    output_path = f'{label}_montaje.png'
    plt.imsave(output_path, image_rgb)

    plt.figure(figsize=(20, 20))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Montaje 2D con máscara – {label}")
    plt.show()



def predict_model(archivo):
    img_dir = Path('imagenes')
    output_dir = Path('imagenes_predicciones')
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob(f'sub{archivo:04d}_slice*.jpg'))

    trained_model = YOLO('runs_yolov11/tumor_detector3/weights/best.pt')

    CONFIDENCE_THRESHOLD = 0.70

    imgs_con_predicciones = []
    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        result = trained_model(img_path)[0]

        boxes = result.boxes
        if boxes is not None and boxes.conf is not None:
            for box, conf, cls in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):
                if conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{result.names[int(cls)]}: {conf:.2f}"
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_bgr, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), img_bgr)

        imgs_con_predicciones.append(img_bgr)

    max_cols = 9
    num_imagenes = len(imgs_con_predicciones)

    if num_imagenes > 0:
        num_filas = (num_imagenes + max_cols - 1) // max_cols
        num_cols = min(num_imagenes, max_cols)

        altura, ancho, canales = imgs_con_predicciones[0].shape

        montaje_matriz = np.ones((num_filas * altura, num_cols * ancho, canales), dtype=np.uint8) * 255

        for i, img in enumerate(imgs_con_predicciones):
            fila = i // max_cols
            columna = i % max_cols

            img_redimensionada = cv2.resize(img, (ancho, altura))

            x_offset = columna * ancho
            y_offset = fila * altura

            montaje_matriz[y_offset:y_offset + altura, x_offset:x_offset + ancho] = img_redimensionada

        out_path = f'montaje_matriz_predicciones_sub{archivo:04d}.png'
        cv2.imwrite(out_path, montaje_matriz)

        plt.figure(figsize=(20, 20))
        plt.imshow(cv2.cvtColor(montaje_matriz, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Montaje en Matriz ({num_filas}x{num_cols}) con Predicciones YOLOv11")
        plt.show()

    else:
        print("No se encontraron imágenes para crear el montaje en matriz.")


archivo = 3
ground_truth(archivo)
predict_model(archivo)
