#!/usr/bin/env bash

# This is a very simple installation script for the rapture AI server.
# Note that rapture AI server only runs on Linux.

# Creates a folder "rapture-AIServer" in your home director and copies any files required into that folder.
mkdir ~/rapture-AIServer
cp -r ./ ~/rapture-AIServer
cd ~/rapture-AIServer

# Sets up a conda enviroment
conda create --prefix ./env flask pytorch sentencepiece sentence-transformers protobuf
conda activate ./env

# The following packages are currently not supported by conda and need to be installed via PIP.
pip install git+https://github.com/huggingface/transformers.git accelerate bitsandbytes bert-extractive-summarizer

# To run the rupture AI server
# go the "rapture-AIServer" folder: `cd ~/rapture-AIServer`
# activate the enviroment: `conda activate ./env`
# run `python3 server.py`
