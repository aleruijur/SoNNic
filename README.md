# SoNNic The Convolutional Network
SoNNic is an AI that can play Sonic The Hedgehog (Sega Master System) on BizHawk emulator using real time CNNs.
This repository is part of a Universidad de Sevilla's final degree project.

<p align="center">
  <img src="./demo.gif"/ width="260"> 
  <img src="./predict.gif"/ width="260">
</p>

- [Watch the AI playing](https://www.youtube.com/watch?v=Sx6vTaZBBg0)
- [Final Degree Project document](https://drive.google.com/)

## Set-up environtment

To run this project, you need **Python 3** and **Bizhawk** emulator.

### Install 64-bit Python 3
This project was written for [Python 3.9](https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe). Tensorflow requires **64-bit Python**.

### Install Python Dependencies
The following Python **dependencies** need to be installed.

- Tensorflow 2.7.0
- Keras 2.7.0
- Pillow
- matplotlib
- mkdir_p
- h5py

### (Optional) Install CUDA and cuDNN
Although you can run Tensorflow on CPU, I'll recommend you to download and install [TensorFlow GPU](https://www.tensorflow.org/install/gpu) dependencies too.

|TensorFlow|Python|cuDNN|CUDA|
|----------|------|-----|----|
|2.7.0|3.5 to 3.9|7.6|10.1|

- [Get CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
- [Get cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (you will need an nvidia account to access)

### Get BizHawk emulator

This project contains *LUA script* files ready to run on BizHawk emulator (tested on version 2.7). To get BizHawk you first need to install the [prerequisites](https://github.com/TASVideos/BizHawk-Prereqs/releases/tag/2.4.8_1). Then you can download [BizHawk](https://github.com/TASEmulators/BizHawk/releases/tag/2.7) and unzip it to any directory.

>You will also need a Sonic The Hedgehog ROM of the Sega Master System version to run on BizHawk emulator.

## Usage Instructions
You must run .py files directly from console and .lua files from BizHaw Lua Console. You can find all lua files on `scripts` folder.

### Running the Project
These instructions can be used to run the project using the `level1` or `level3` model.

1. Run `predict-server.py` using Python or Docker - this starts a server on port `36296` which actually runs the model.
    - You must specify the model you want to run. Use `level1` or `level3` as first parameter.
    - You can pass a `--cpu` to force Tensorflow to run on the CPU.
2. Open BizHawk and load a Sonic The Hedgehog ROM.
3. Open the BizHawk Lua console (Tools > Lua Console).
4. Load `ScriptHawk.lua` and click the hitbox checkbox.
5. Load `Play.lua`

### Generate your own training data
The first thing you need to train your model is training data. You can generate training data using `RecordInput.lua`.
1. Open BizHawk and load a Sonic The Hedgehog ROM.
2. Open the BizHawk Lua console (Tools > Lua Console).
4. Load `ScriptHawk.lua` and click the hitbox checkbox.
3. Load `RecordInput.lua`
4. Play for a while

A new folder will be created on `recordings`. A screenshot will be stored every frame after you start playing the game with the binary values stored on `inputs.txt`

### Training the Model on Recordings
Once you have generated new recording, you probably want to try retraining the weights based off your recordings. To train a new model, run `train.py [model]`. You can also use `--cpu` to force it to use the CPU. Your trained model will be stored on `weights/[model].hdf5`

### Play your trained model on a race
You can load the levels savestates from `states/Acto1.state` or `states/Acto3.state` to test your new trained model.
Remember to launch `predict-server.py` first and load `Play.lua` from Lua console.

### Train AI to play on another level
You can use `RecordInput.lua` to generate training data for another level. Even for another game!
Remember to use a different name as parameter when you train your model with `train.py`.

## Reference Projects
- [NeuralKart](https://github.com/rameshvarun/NeuralKart) - This project was forked from rameshvarun real time Mario Kart AI.
- [ScriptHawk](https://github.com/isotarge/ScriptHawk) - A collection of Lua scripts for BizHawk providing tools to assist TASing.
