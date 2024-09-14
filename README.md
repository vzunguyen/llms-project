# Project Brief
This project is based on the workshop [Building LLMs from the Ground Up: A 3-hour Coding Workshop](https://magazine.sebastianraschka.com/p/building-llms-from-the-ground-up). It is my first exploration into building a language model from scratch.

## Project Overview
The goal of this project is to understand the inner workings of large language models (LLMs) by implementing one from the ground up.

## Project Progress
- Data Processing
- Data Set, Data Load
- GPT architecture
    - Multihead Attention
    - Layer Norm
    - GELU
    - Feed Forward
    - Transformer Block
- Generating new text (Untrained)
- Train model
- Weightloading
- Finetuning

## Technology being used in LLM Project
- LitGPT

## How to run?
**Step 1:** Download [python](https://www.python.org/downloads/) 🫡
**Step 2:** Download [Miniforge3](https://github.com/conda-forge/miniforge)
    For MacOS/Linux, Homebrew has Miniforge3. Run:
```
brew install --cask miniforge
```
**Step 3:** Run:
```
conda config --set solver libmamba
```
**Step 4:** Create new virtual environment:
```
conda create -n LLMs python=3.10
```
To activate the environment:
```
conda activate LLMs
```
**Step 5:** Install required Python libraries:
    - Download requirements.txt
    - Run:
```
pip install -r requirements.txt
```

# References
[Building LLMs from the Ground Up: A 3-hour Coding Workshop](https://magazine.sebastianraschka.com/p/building-llms-from-the-ground-up)

Apart from "Building LLMs from the Ground Up: A 3-hour Coding Workshop", [Create a Large Language Model from Scratch with Python – Tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8) was also used to help with developing the project.

*This project is developed by Zu*