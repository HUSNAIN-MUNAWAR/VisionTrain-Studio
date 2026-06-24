# Troubleshooting

## Backend does not start

Run setup first:

```powershell
scripts\setup_windows.bat
```

Then start the app:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_visiontrain.ps1
```

Check logs:

```text
backend/logs/backend_runtime.log
backend/logs/frontend_runtime.log
```

## GPU selected but blocked

Run:

```powershell
scripts\check_gpu.bat
```

Common causes:

- NVIDIA driver is missing or old.
- CPU PyTorch was installed instead of CUDA PyTorch.
- CUDA build does not match your driver.

## TensorRT export disabled

VisionTrain does not install TensorRT automatically. Install NVIDIA TensorRT separately, confirm Python can `import tensorrt`, and confirm CUDA is available.

## Dataset validation fails

Open the Dataset page and check the exact missing paths. The validator returns errors and recommended fixes for YOLO, ImageFolder, semantic masks, COCO, anomaly, and OCR placeholder datasets.
