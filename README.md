# mlinstruct
Python module providing a naive abstraction for framework-agnostic model training.

Currently supports only PyTorch

## [TODO] Docs

The aim of this project is to set up abstracted trainer classes that can be used to train deep learning models independent of the framework used.

TODO: mlinstruct component documentation

## Pending Tasks
- Technical:
    - [X] Refactor Trainer submodule
    - [X] Refactor Evaluator submodule
    - [ ] Support loading and storing of models in ONNX format.
    - [ ] Add Kfold Cross Validation Trainer
    - [ ] Add GAN Trainer
    - [ ] Investigate popular training strategies for new trainer ideas

- Organization
    - [ ] Prepare documentation
    - [ ] Add unit tests for Trainer
    - [ ] Investigate Pydantic for data type validation
