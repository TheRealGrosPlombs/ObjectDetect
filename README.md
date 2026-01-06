# Object Detect  

AprilTag Detection Python Script to track an object using a wifi web cam.  
Uses Python 3.13 or less  

## Installation no CUDA  
```
py -3.13 -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## Installation with CUDA  

PyTorch with CUDA support should be installed before other dependencies  

Create your virtual environment:
```
py -3.13 -m venv .venv
./.venv/Scripts/Activate.ps1
```

Check if your have an NVIDIA GPU with correct drivers installed  
```
nvidia-smi
```
You should see something like this  
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 571.96                 Driver Version: 571.96         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
...
```
PIP install the correct [PyTorch library](https://pytorch.org/get-started/locally/) according to your CUDA Vesrion. In this example it is CUDA 12.8.
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install other dependencies:  
```
pip install -r requirements.txt
```

## Usage  
Assuming your camera has been calibrated beforehand with open CV   and accessing high quality stream suing channel 0
```
py detect_tag.py
```

To specify your camera with lower quality channel, use a command line argument, example: rtsp://thingino:thingino@192.168.0.xx:554/ch1  
```
py detect_tag.py <your_ip_camear_url>/ch1
```


## Calibration  
If displayed coordinates are not centered on the AprilTag, camera calibration is required  

TODO...