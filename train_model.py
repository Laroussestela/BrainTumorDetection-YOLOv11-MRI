from ultralytics import YOLO

data_yaml = 'data.yaml'

model = YOLO('yolo11n.pt')

results = model.train(
    data=data_yaml,
    epochs=100,
    patience=7,
    project='runs_yolov11',
    name='tumor_detector'
)
