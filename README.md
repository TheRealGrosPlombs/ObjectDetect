# Object Detect  

AprilTag Detection Python Script to track an object using a wifi web cam.  
Uses Python 3.13 or less  

## Installation  
```
py -3.13 -m venv .venv
./.venv/Scripts/Activate.ps1
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
TODO...