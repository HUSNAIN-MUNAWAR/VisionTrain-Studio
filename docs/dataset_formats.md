# Dataset Formats

## YOLO detection / segmentation / pose

```text
dataset/
  dataset.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

`dataset.yaml` should define train/val paths and class names. Pose datasets must be valid Ultralytics pose datasets.

## Classification folders

```text
dataset/
  train/
    class_1/
    class_2/
  val/
    class_1/
    class_2/
```

## Semantic segmentation masks

```text
dataset/
  images/
    train/
    val/
  masks/
    train/
    val/
```

Image and mask filenames should match by stem, for example `frame_001.jpg` and `frame_001.png`.

## COCO detection

```text
dataset/
  images/
    train/
    val/
  annotations/
    train.json
    val.json
```

The TorchVision detection starter expects COCO-style boxes and category IDs.

## Anomaly detection

```text
dataset/
  train/
    normal/
  val/
    normal/
    anomaly/
```

The autoencoder learns normal reconstruction and reports validation reconstruction scores.
