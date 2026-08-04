# YOLO26n ONNX setup

CareBot now starts with the YOLO detector by default and expects its model at
`CareBot_V3/models/yolo26n.onnx`. The runtime uses the already-installed
`onnxruntime` package; it does not need PyTorch or `ultralytics` on the robot.

Export the model once on a development computer, then copy the resulting ONNX
file to the robot:

```bash
python -m pip install -U ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=320 simplify=True
scp yolo26n.onnx pi@<robot-ip>:/home/pi/project_demo/09.AI_Big_Model/AI_CarAgent_en/CareBot_V3/models/
```

`imgsz=320` is the recommended first setting for Raspberry Pi latency.  If
person detection is not reliable enough at distance, export at `imgsz=416` or
`640` and compare the on-screen inference time and FPS.

Run the new detector:

```bash
python3 CareBot_Main.py --no-motor
```

Use another valid ONNX location if needed:

```bash
python3 CareBot_Main.py --yolo-model /path/to/yolo26n.onnx
```

The original TensorFlow SSD model is retained for comparison or rollback:

```bash
python3 CareBot_Main.py --detector ssd --no-motor
```
