# Tensorflow Inference

## Overview
mp-tensorflow is a package for secure tensorflow inference that utilizes the MP-SPDZ framework.
For all supported models, computation are preformed in four parts:
1. Pretrained model is downloaded
2. Resource/Dependency collection 
3. Data processing
4. MP-SPDZ compilation

Supported models:
- [SqueezeNet](#squeezenet)
- [ResNet](#squeezenet)

## Requirements
Before running Tensorflow inference, be sure to run the compilation instructions once:
```
Scripts/setup-ssl.sh
make -j 8 tldr
make replicated-ring-party.x
```
Python version must be between 3.5 - 3.7.
Additional dependencies may be installed through `requirements.txt`:
```
cd TensorflowInf
pip install -r requirements.txt
```
**WARNING:** installing through `requirements.txt` may downgrade some Python libraries. For that reason, it is recommended to use [virtualenv](https://virtualenv.pypa.io/en/latest/) to isolate pip installs.

## SqueezeNet

### Usage
```
cd TensorflowInf/SqueezeNet
./tf-inference.sh -i SampleImages/sample_image_1.JPEG -n 1
```

### Options
**-i** *<INPUT_IMAGE_FILE>*
    Path to image file to run computation on. This may be relative or absolute path.
**-n** *<NUMBER_OF_THREADS>*
    Number of threads for compilation

## ResNet

### Usage
```
cd TensorflowInf/ResNet
./tf-inference.sh -i SampleImages/sample_image_1.JPEG -n 1
```

### Options
**-i** *<INPUT_IMAGE_FILE>*
    Path to image file to run computation on. This may be relative or absolute path.
**-n** *<NUMBER_OF_THREADS>*
    Number of threads for compilation
