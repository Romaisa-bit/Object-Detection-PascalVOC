# Object Detection using Pascal VOC 2012

This project trains an object detection/classification model using the Pascal VOC 2012 dataset and TensorFlow. The model uses MobileNetV2 as a feature extractor and can be converted into TensorFlow Lite format for deployment on mobile and edge devices.

## Features

* Pascal VOC 2012 Dataset
* MobileNetV2 Transfer Learning
* TensorFlow/Keras Training
* TensorFlow Lite Conversion
* Model Checkpointing
* Dataset Visualization

## Dataset Classes

person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, diningtable, pottedplant, sofa, tvmonitor

## Installation

```bash
pip install -r requirements.txt
```

## Download Dataset

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
```

## Training

```bash
python train_model.py
```

## Convert Model to TensorFlow Lite

```bash
python convert_to_tflite.py
```

## Output Files

* object_detection_model.keras
* object_detection_model.tflite

## Technologies

* TensorFlow
* OpenCV
* NumPy
* Matplotlib
* MobileNetV2

## License

MIT License
