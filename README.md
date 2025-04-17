# Carla-Opencda-with-webrtc
A WebRtc bridge between Carla and Opencda

## ENV -- 4.17
Now you can create your conda environment by `conda env create -f environment.yml` now!
I think this file has included all dependencies you'll need.

## Local installation
### REPO Installation
```
git clone https://github.com/cyKkk0/Carla-Opencda-with-webrtc.git
```
### Carla
We use Carla-0.9.15 and it's **not** recommanded to build from source.You can download it from [here](https://tiny.carla.org/carla-0-9-15-linux)

Also, additional map is needed.You can download from [here](https://tiny.carla.org/additional-maps-0-9-15-linux)

### Opencda
We did some changes in original repo and you should just clone this repo instead of downloading from the official site.
But you still need to install the requirements.
In `Opencda/`
```
conda env create -f environment.yml
conda activate opencda
python setup.py develop
```
After dependencies are installed, we need to install the CARLA python library into opencda conda environment. You can do this by running this script:
```
export CARLA_HOME=/path/to/your/CARLA_ROOT
export CARLA_VERSION=0.9.11 #or 0.9.12 depends on your CARLA
. setup.sh
```
If everything works correctly, you will see a cache folder is created in your OpenCDA root dir, and the terminal shows “Successful Setup!”. To double check the carla package is correctly installed, run the following command and there should be no error.
```
If everything works correctly, you will see a cache folder is created in your OpenCDA root dir, and the terminal shows “Successful Setup!”. To double check the carla package is correctly installed, run the following command and there should be no error.
```
### Pytorch and Yolov5
You need these two, or this repo is meaningless.
The command belows shows an example of installing pytorch v1.8.0 with cuda 11.1 in opencda environment.
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
After pytorch installation, install the requirements for Yolov5.
```
pip -r https://raw.githubusercontent.com/ultralytics/yolov5/refs/tags/v6.2/requirements.txt
```
**The up-to-date version'requirements conflict.The best fit one is v6.2**
### SUMO
SUMO installation is only required for the users who require to conduct co-simulation testing and use future release of SUMO-only mode.
You can install SUMO directly by apt-get:
```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
After that, install the traci python package.
```
pip install traci
```
Finally, add the following path to your ~/.bashrc:
```
export SUMO_HOME=/usr/share/sumo
```
### aiortc
You can install this through pip.
```
pip install aiortc
```
You'd better do some changes in `aiortc/codecs/h264.py` and `aiortc/codecs/vpx.py`:
increase the `DEFAULT_BITRATE`, `MIN_BITRATE`, `MAX_BITRATE` to maybe 10Mbps or bigger...
### av
Though `av` will be downloaded when you install `aiortc`, on my device, this version will lead to the dumps when running carla and opencda.
My solution is use conda to install this library.If you don't have such problem, skip this.
```
conda install av=10.0.0 -c conda-forge
```

Now hope you can run the repo successfully :).

## Quick start
You can test if the env is ok first.
In OpenCDA/, run the following commands.
### 1. Two-lane highway test
```
python opencda.py -t single_2lanefree_carla -v 0.9.12
```
### 2. Town06 test (Pytorch required)
```
python opencda.py -t single_town06_carla  -v 0.9.12 --apply_ml
```
### 3. Town06 Co-simulation test (Pytorch and Sumo required)
```
python opencda.py -t single_town06_cosim  -v 0.9.12 --apply_ml
```
### 4. Town06 with WebRTC
To enable the WebRTC by `--webrtc`
Current is only available in single_town06_carla scenario.
```
python opencda.py -t single_town06_carla  -v 0.9.12 --apply_ml --webrtc
```
