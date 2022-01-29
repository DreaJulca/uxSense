# uxSense

## Setup steps
To use the interface, you will need Node.js and React.

To run the back-end models yourself, you will need MongoDB, Python 3, [Tensorflow](https://www.tensorflow.org/install) [(GPU)](https://www.tensorflow.org/install/gpu), [caffe2](https://caffe2.ai/docs/getting-started.html), and [PyTorch](https://pytorch.org/get-started/locally/).

These last three will require [CUDA Toolkit](https://docs.nvidia.com/cuda/index.html), [cuDNN](https://developer.nvidia.com/cudnn), [CMake](https://cmake.org/download/), and---at least on a windows machine---[Cygwin](https://www.mingw-w64.org/downloads/#cygwin), ZLIB, boost, BLAS, and Visual Studio 2019 with C++ build tools for 2015-2022. [Ninja](https://ninja-build.org/) is the default build system but in my experience it's probably more trouble than it's worth, so if you opt not to use it, make sure you set the appropriate flags in your build configs. 

