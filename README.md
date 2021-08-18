# N-ZeroX
(Neural)-Zero X is an AI that can play F-Zero X on BizHawk emulator using real time CNNs.
This repository is part of a Universidad de Sevilla's final degree project.

<p align="center">
  <img src="./train.gif"/>
  <img src="./demo.gif"/>
</p>

- [Watch the AI playing](https://www.youtube.com/)
- [Final Degree Project document](https://drive.google.com/)

## Set-up environtment

To run this project, you need **Python 3** and **Bizhawk** emulator.

### Install 64-bit Python 3
This project was written for [Python 3.7](https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe). Tensorflow requires **64-bit Python**.

### Install Python Dependencies
The following Python **dependencies** need to be installed.

- Tensorflow 2.2.0
- Keras 2.3.1
- Pillow
- matplotlib
- mkdir_p
- h5py

### (Optional) Install CUDA and cuDNN
Although you can run Tensorflow on CPU, I'll recommend you to download and install [TensorFlow GPU](https://www.tensorflow.org/install/gpu) dependencies too.

|TensorFlow|Python|cuDNN|CUDA|
|2.2.0|3.5 to 3.8|7.6|10.1|

- [Get CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
- [Get cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (you will need an nvidia account to access)

### Get BizHawk emulator

This project contains *LUA script* files ready to run on BizHawk emulator (tested on version 2.6.2). To get BizHawk you first need to install the [prerequisites](https://github.com/TASVideos/BizHawk-Prereqs/releases/tag/2.4.8_1). Then you can download [BizHawk](https://github.com/TASVideos/BizHawk/releases/tag/2.6.2) and unzip it to any directory.

>You will also need a F-Zero X ROM to run on BizHawk emulator.

### Download Pre-trained Weights and Recordings
Download this data to run the demo. You can also download my recordings to train the models by yourself. These should be *unzipped* into the folder of this repository.

- [Save States](https://drive.google.com/) - LUA scripts will access the *saved states* on `states/[file].state`.
- [Weights](https://drive.google.com/) - Python scripts will access the *trained models* on `weights/[model].hdf5`
- [Recordings (Optional)](https://drive.google.com/) - The recordings should be accessible as `recordings/[recording]/[frame].png`.

## Usage Instructions
### Running the Demo
These instructions can be used to run a demo of three tracks that the AI performs well on.

1. Download the save states and pre-trained model.
2. Run `predict-server.py` using Python - this starts a server on port `36296` which actually runs the model.
    - You must specify the model you want to run. Use `demo` as first parameter.
    - You can pass a `--cpu` to force Tensorflow to run on the CPU.
3. Open BizHawk and load a F-Zero X ROM.
4. Turn off messages (View > Display Messages).
    - You don't have to do this, but they get in the way.
4. Open the BizHawk Lua console (Tools > Lua Console).
5. Load `Demo.lua`

This should automatically play Mute City time attack race.  You can hit the arrow keys to manually steer the Blue Falcon. This can be used to demonstrate the AI's stability.

### Generate your own training data


### Training the Model on Recordings
Once you have generated new recording, you probably want to try retraining the weights based off your recordings. To train a new model, run `train.py [model]`. You can also use `--cpu` to force it to use the CPU. Your trained model will be stored on `weights/[model].hdf5`

### Play your trained model on a race


### Train AI to play on another track


## Reference Projects

- [NeuralKart](https://github.com/rameshvarun/NeuralKart) - This project was forked from rameshvarun real time Mario Kart AI.
- [TensorKart](https://github.com/kevinhughes27/TensorKart) - The first MarioKart deep learning project, used for reference.
