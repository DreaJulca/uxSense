# hcEye
Human-Computer Interaction via Computer Vision

# REQUIREMENTS
* node.js
* python (>=3.6.5)
    * Tensorflow is required
    * PyTorch and caffe2 are highly recommended (and necessary for 3D pose coordinates)
    * Either tensorflow-probability or tensorflow-probability-gpu is required for action prediction
* R (>=3.4.1)
* ffmpeg (optional, but we recommend downloading and installing it from https://ffmpeg.org/ and adding it to your PATH if you have not already done so)

# Node
Just install package dependencies, then start: 
```
npm i
npm start
```

# Models

Set up to run models with
```
python -m pip install -r requirements.txt
```

If you are using windows, you must first install https://github.com/philferriere/cocoapi: 

```
python -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

Note that this requires Visual Studio Build Tools (https://visualstudio.microsoft.com/downloads/)

2D human pose detection based on https://github.com/ildoonet/tf-pose-estimation, and 3D human pose detection on https://github.com/facebookresearch/VideoPose3D.

Both depend on tensorflow, and the latter depends on pytorch+caffe2; if not already configured, please follow installation instructions at https://www.tensorflow.org/install and https://pytorch.org/get-started/locally/ + https://caffe2.ai/docs/getting-started.html, respectively.

The latter also depends on Detectron (https://github.com/facebookresearch/Detectron); if using windows, you will need to install make (see http://gnuwin32.sourceforge.net/packages/make.htm) and then modify the Cython files--particularly those in detectron/utils, where you will need to add ```.astype('int32')``` to the end of one of your cdefs (specifically, ```cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1].astype('int32')```). 

If using tensorflow-gpu and not already configured, remember to first follow CUDA setup instructions at https://developer.nvidia.com/cuda-10.0-download-archive after ensuring that your GPU driver is up to date. You may need to uninstall Visual Studio Build Tools. Then, install cuDNN (https://developer.nvidia.com/cudnn) and add to your path. 
