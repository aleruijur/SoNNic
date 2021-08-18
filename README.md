# N-ZeroX
(Neural)-Zero X is an AI that can play F-Zero X on BizHawk emulator using real time CNNs.
This repository is part of a Universidad de Sevilla's final degree project.

<p align="center">
  <img src="./train.gif"/>
</p>

- [Watch the AI playing](https://www.youtube.com/)
- [Final Degree Project document](https://drive.google.com/)

## Set-up

To run this project, you need Python 3 and Bizhawk emulator.

### Install 64-bit Python 3
This project was written for [Python 3.7](https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe). Tensorflow requires 64-bit Python.

### Install Python Dependencies
The following Python dependencies need to be installed.

- Tensorflow 2.2.0
- Keras 2.3.1
- Pillow
- matplotlib
- mkdir_p
- h5py

### Get BizHawk

This project contains LUA script files ready to run on BizHawk emulator (tested on version 2.6.2). To get BizHawk you first need to install the [prerequisites](https://github.com/TASVideos/BizHawk-Prereqs/releases/tag/2.4.8_1). Then you can download [BizHawk](https://github.com/TASVideos/BizHawk/releases/tag/2.6.2) and unzip it to any directory.

### Download Our Pre-trained Weights and Recordings
These should be unzipped into the folder of the repo.

- [Save States](https://drive.google.com/open?id=0B7KSCOuXHAaQaGNDWEI2MlBSRDQ) - The states should be accessible as `states/[mode]/[track].state`.
- [Weights](https://drive.google.com/open?id=0B7KSCOuXHAaQQUY3V2dqQjNNbXM) - The weights should be accessible as `weights/[track].hdf5`
- [Recordings (Optional)](https://drive.google.com/open?id=0B7KSCOuXHAaQSHFLRFpCQTBVemM) - The recordings should be accessible as `recordings/[track]/[mode]/[recording]/[frame].png`.

## Usage Instructions
### Running a Live Demo
These instructions can be used to run a demo of three tracks that the AI performs well on.

1. Download the save states and pre-trained weights.
2. Run `predict-server.py` using Python 3 - this starts a server on port `36296` which actually runs the model.
    - You can pass a `--cpu` to force Tensorflow to run on the CPU.
3. Open BizHawk and Load the MarioKart 64 ROM.
4. Turn off messages (View > Display Messages).
    - You don't have to do this, but they get in the way.
4. Open the BizHawk Lua console (Tools > Lua Console).
5. Load `Demo.lua`

This should automatically play three tracks in a loop. You can hit `Esc` to switch to the next track. You can also hit the arrow keys to manually steer the player. This can be used to demonstrate the AI's stability.

Note that the clipboard is used to pass frames from the emulator to the Python script. It's a hack, but it seems to work - just don't try to copy or paste anything while the scripts are running.

### Run the AI on another Track
Once you have the demo working, you can use these instructions to play on other tracks. Note that you can only play on a track if there are weights trained for it.

First, navigate to another track from the menu, or use one of our save states (File > Load State > Load Named State). These states are set to be the frame after the race starts. Then load `Play.lua` from the Lua console.

### Training the Model on Recordings
Once you have the AI running, you probably want to try retraining the weights based off our recordings. First download our weights from the link above, then run `train.py [track]`. You can also use `--cpu` to force it to use the CPU.

### Creating new Recordings from the Search AI
Load a state and then load `SearchAI.lua` in order to generate a recording using the search AI. Recordings consist of a series of frames and a `steering.txt` file that contains the recorded steering values.

### Running the Iterative Improvement Loop
As mentioned in the paper, we ran an iterative improvement loop that swaps between playing and generating new recordings. To bootstrap the process, you must first generate a recording using the search AI and create an initial weights file using `train.py`. Now start `predict-server.py` using the `--cpu` flag (so that you can train on the GPU).

Now you can load a state and run `PlayAndSearch.lua` which alternates between playing and searching. It retrains every other run. You probably need to edit the code that calls `train.py` on line 90 so that it works in your environment.

## Other Projects + Links

- [TensorKart](https://github.com/kevinhughes27/TensorKart) - The first MarioKart deep learning project, which we started from as our baseline.
- [Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](https://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf) - The idea for using a search-based AI for teaching the Convnet AI came from this paper.
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf) - The DAGGER algorithm was first introduced in this paper.
- [MarioKart 64 NEAT](https://www.youtube.com/watch?v=tmltm0ZHkHw) - This AI uses the NEAT algorithm to genetically evolve a shallow neural network
- [weatherton/BizHawkMarioKart64](https://github.com/weatherton/BizHawkMarioKart64) - Some MarioKart 64 scripts which we used as a reference for memory locations.
