# object_detection
## Requirements
1. (Visual Studio 2019 with C++ Build Tools is required.
https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16)
2. ( Visual C++ 2015 build is required 
https://go.microsoft.com/fwlink/?LinkId=691126)
## List of CUDA enabled devices https://developer.nvidia.com/cuda-gpus
3. Install NVIDIA DRIVER https://www.nvidia.com/Download/index.aspx
4. Install Install CUDA TOOLKIT v11.1 https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_456.81_win10.exe
5. Download cuDNN https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.1_20201106/cudnn-11.1-windows-x64-v8.0.5.39.zip
6. Download Anaconda https://www.anaconda.com/products/individual
## Create and activate virtual environment.
```
conda create -n tensorflow pip python=3.8

conda activate tensorflow
```
## install tensorflow gpu
```
pip install tensorflow-gpu
```
```
python
  >>> import tensorflow as tf
  >>> print(tf.__version__)
  >>> exit()
```
```
mkdir TensorFlow
cd C:\TensorFlow
```
## Download Model
```
conda install -c anaconda git
git clone https://github.com/tensorflow/models.git
cd models\research
protoc object_detection\protos\*.proto --python_out=.
```
Close CMD.
## Open Anaconda promt
```
conda activate tensorflow
python -m pip install --upgrade pip
pip install cython
conda install -c anaconda protobuf
pip install pillow
pip install lxml
pip install Cython
pip install contextlib2
pip install jupyter
pip install matplotlib
pip install pandas
pip install opencv-python
```
install pycocotools using following command
```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI 
```
```
cd C:\TensorFlow\models\research
```
```
copy object_detection\packages\tf2\setup.py .
python -m pip install .
```
## check if everything is working perfectly.
```
python object_detection\builders\model_builder_tf2_test.py
```
## Export pre-trained test.record and train.record
```
cd C:\TensorFlow\scripts\preprocessing
python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\train -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\train.record
python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\test -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\test.record
```
## place the downloaded model in pre-trained model
Download ssd mobilenet models from here.
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
copy and edit pipeline.config to models/my_ssd_mobilenet_v2_fpnlite
```
Line 3. Change num_classes to the number of classes your model detects.3
Line 135. Change batch_size according to available memory (Higher values require more memory and vice-versa). I changed it to:
batch_size: 5
Line 165. Change fine_tune_checkpoint to:
fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"
Line 171. Change fine_tune_checkpoint_type to:
fine_tune_checkpoint_type: "detection"
Line 175. Change label_map_path to:
label_map_path: "annotations/label_map.pbtxt"
Line 177. Change input_path to:
input_path: "annotations/train.record"
Line 185. Change label_map_path to:
label_map_path: "annotations/label_map.pbtxt"
Line 189. Change input_path to:
input_path: "annotations/test.record"
```
```
cd C:\TensorFlow\workspace\training_demo
python model_main_tf2.py --model_dir=models\my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models\my_ssd_mobilenet_v2_fpnlite\pipeline.config
```
## monitoring training with tensorboard- open another anaconda promt
```
conda activate tensorflow
cd C:\TensorFlow\workspace\training_demo
tensorboard --logdir=models\my_ssd_mobilenet_v2_fpnlite
```
## Exporting the Inference Graph
```
cd C:\TensorFlow\workspace\training_demo

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir .\models\my_ssd_mobilenet_v2_fpnlite\ --output_directory .\exported-models\my_mobilenet_model
```


## Evaluating the Model
```
cd C:\TensorFlow\workspace\training_demo
python model_main_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --model_dir models\my_ssd_mobilenet_v2_fpnlite --checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --alsologtostderr
```

## Using the model
```
cd C:\TensorFlow\workspace\training_demo
python TF-image-od.py
```
### Step 2. Build TensorFlow From Source
To convert the frozen graph we just exported into a model that can be used by TensorFlow Lite, it has to be run through the TensorFlow Lite Optimizing Converter (TOCO). Unfortunately, to use TOCO, we have to build TensorFlow from source on our computer. To do this, we’ll create a separate Anaconda virtual environment for building TensorFlow. 

This part of the tutorial breaks down step-by-step how to build TensorFlow from source on your Windows PC. It follows the [Build TensorFlow From Source on Windows](https://www.tensorflow.org/install/source_windows) instructions given on the official TensorFlow website, with some slight modifications. 

This guide will show how to build either the CPU-only version of TensorFlow or the GPU-enabled version of TensorFlow v1.13. If you would like to build a version other than TF v1.13, you can still use this guide, but check the [build configuration list](https://www.tensorflow.org/install/source_windows#tested_build_configurations) and make sure you use the correct package versions. 

**If you are only building TensorFlow to convert a TensorFlow Lite object detection model, I recommend building the CPU-only version!** It takes very little computational effort to export the model, so your CPU can do it just fine without help from your GPU. If you’d like to build the GPU-enabled version anyway, then you need to have the appropriate version of CUDA and cuDNN installed. [The TensorFlow installation guide](https://www.tensorflow.org/install/gpu#windows_setup) explains how to install CUDA and cuDNN. Check the [build configuration list](https://www.tensorflow.org/install/source_windows#tested_build_configurations) to see which versions of CUDA and cuDNN are compatible with which versions of TensorFlow.

**If you get any errors during this process, please look at the [FAQ section](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#frequently-asked-questions-and-common-errors) at the bottom of this guide! It gives solutions to common errors that occur.**

#### Step 2a. Install MSYS2
MSYS2 has some binary tools needed for building TensorFlow. It also automatically converts Windows-style directory paths to Linux-style paths when using Bazel. The Bazel build won’t work without MSYS2 installed! 

First, install MSYS2 by following the instructions on the [MSYS2 website](https://www.msys2.org/). Download the msys2-x86_64 executable file and run it. Use the default options for installation. After installing, open MSYS2 and issue:

```
pacman -Syu
```

<Picture of MSYS2 shell to be added here>

After it's completed, close the window, re-open it, and then issue the following two commands:

```
pacman -Su
pacman -S patch unzip
```

<p align="center">
   <img src="doc/MSYS_window.png">
</p>

This updates MSYS2’s package manager and downloads the patch and unzip packages. Now, close the MSYS2 window. We'll add the MSYS2 binary to the PATH environment variable in Step 2c.

#### Step 2b. Install Visual C++ Build Tools 2015
Install Microsoft Build Tools 2015 and Microsoft Visual C++ 2015 Redistributable by visiting the [Visual Studio older downloads](https://visualstudio.microsoft.com/vs/older-downloads/) page. Click the “Redistributables and Build Tools” dropdown at the bottom of the list.  Download and install the following two packages:

* **Microsoft Build Tools 2015 Update 3** - Use the default installation options in the install wizard. Once you begin installing, it goes through a fairly large download, so it will take a while if you have a slow internet connection. It may give you some warnings saying build tools or redistributables have already been installed. If so, that's fine; just click through them.
* **Microsoft Visual C++ 2015 Redistributable Update 3** – This may give you an error saying the redistributable has already been installed. If so, that’s fine.

Restart your PC after installation has finished.

#### Step 2c. Update Anaconda and create tensorflow-build environment
Now that the Visual Studio tools are installed and your PC is freshly restarted, open a new Anaconda Prompt window. First, update Anaconda to make sure its package list is up to date. In the Anaconda Prompt window, issue these two commands: 

```
conda update -n base -c defaults conda
conda update --all
```

The update process may take up to an hour, depending on how it's been since you installed or updated Anaconda. Next, create a new Anaconda virtual environment called “tensorflow-build”. We’ll work in this environment for the rest of the build process. Create and activate the environment by issuing:

```
conda create -n tensorflow-build pip python=3.6
conda activate tensorflow-build
```

After the environment is activated, you should see (tensorflow-build) before the active path in the command window. 

Update pip by issuing:

```
python -m pip install --upgrade pip
```

We'll use Anaconda's git package to download the TensorFlow repository, so install git using:

```
conda install -c anaconda git
```

Next, add the MSYS2 binaries to this environment's PATH variable by issuing:

```
set PATH=%PATH%;C:\msys64\usr\bin
```

(If MSYS2 is installed in a different location than C:\msys64, use that location instead.) You’ll have to re-issue this PATH command if you ever close and re-open the Anaconda Prompt window. 

#### Step 2d. Download Bazel and Python package dependencies
Next, we’ll install Bazel and some other Python packages that are used for building TensorFlow. Install the necessary Python packages by issuing: 

```
pip install six numpy wheel
pip install keras_applications==1.0.6 --no-deps
pip install keras_preprocessing==1.0.5 --no-deps
```

Then install Bazel v0.21.0 by issuing the following command. (If you are building a version of TensorFlow other than v1.13, you may need to use a different version of Bazel.)

```
conda install -c conda-forge bazel=0.21.0
```

#### Step 2d. Download TensorFlow source and configure build
Time to download TensorFlow’s source code from GitHub! Issue the following commands to create a new folder directly in C:\ called “tensorflow-build” and cd into it:

```
mkdir C:\tensorflow-build
cd C:\tensorflow-build
```

Then, clone the TensorFlow repository and cd into it by issuing: 

```
git clone https://github.com/tensorflow/tensorflow.git 
cd tensorflow 
```

Next, check out the branch for TensorFlow v1.13: 

```
git checkout r1.13
```

The version you check out should match the TensorFlow version you used to train your model in [Step 1](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#step-1-train-quantized-ssd-mobilenet-model-and-export-frozen-tensorflow-lite-graph). If you used a different version than TF v1.13, then replace "1.13" with the version you used. See the [FAQs section](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#how-do-i-check-which-tensorflow-version-i-used-to-train-my-detection-model) for instructions on how to check the TensorFlow version you used for training.

Next, we’ll configure the TensorFlow build using the configure.py script. From the C:\tensorflow-build\tensorflow directory, issue:

```
python ./configure.py
```

This will initiate a Bazel session. As I mentioned before, you can build either the CPU-only version of TensorFlow or the GPU-enabled version of TensorFlow. If you're only using this TensorFlow build to convert your TensorFlow Lite model, **I recommend building the CPU-only version**. If you’d still like to build the GPU-enabled version for some other reason, then you need to have the appropriate version of CUDA and cuDNN installed. This guide doesn't cover building the GPU-enabled version of TensorFlow, but you can try following the official build instructions on the [TensorFlow website](https://www.tensorflow.org/install/source_windows).

Here’s what the configuration session will look like if you are building for CPU only. Basically, press Enter to select the default option for each question.

```
You have bazel 0.21.0- (@non-git) installed. 

Please specify the location of python. [Default is C:\ProgramData\Anaconda3\envs\tensorflow-build\python.exe]: 
  
Found possible Python library paths: 

  C:\ProgramData\Anaconda3\envs\tensorflow-build\lib\site-packages 

Please input the desired Python library path to use.  Default is [C:\ProgramData\Anaconda3\envs\tensorflow-build\lib\site-packages] 

Do you wish to build TensorFlow with XLA JIT support? [y/N]: N 
No XLA JIT support will be enabled for TensorFlow. 

Do you wish to build TensorFlow with ROCm support? [y/N]: N 
No ROCm support will be enabled for TensorFlow. 
  
Do you wish to build TensorFlow with CUDA support? [y/N]: N 
No CUDA support will be enabled for TensorFlow. 
```

Once the configuration is finished, TensorFlow is ready to be bulit!

#### Step 2e. Build TensorFlow package
Next, use Bazel to create the package builder for TensorFlow. To create the CPU-only version, issue the following command. The build process took about 70 minutes on my computer. 

```
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package 
```

Now that the package builder has been created, let’s use it to build the actual TensorFlow wheel file. Issue the following command (it took about 5 minutes to complete on my computer): 

```
bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg 
```

This creates the wheel file and places it in C:\tmp\tensorflow_pkg.

#### Step 2f. Install TensorFlow and test it out!
TensorFlow is finally ready to be installed! Open File Explorer and browse to the C:\tmp\tensorflow_pkg folder. Copy the full filename of the .whl file, and paste it in the following command:

```
pip3 install C:/tmp/tensorflow_pkg/<Paste full .whl filename here>
```

That's it! TensorFlow is installed! Let's make sure it installed correctly by opening a Python shell:

```
python
```

Once the shell is opened, issue these commands:

```
>>> import tensorflow as tf
>>> tf.__version__
```

If everything was installed properly, it will respond with the installed version of TensorFlow. Note: You may get some deprecation warnings after the "import tensorflow as tf" command. As long as they are warnings and not actual errors, you can ignore them! Exit the shell by issuing:

```
exit()
```

With TensorFlow installed, we can finally convert our trained model into a TensorFlow Lite model. On to the last step: Step 3!

### Step 3. Use TOCO to Create Optimzed TensorFlow Lite Model, Create Label Map, Run Model
Although we've already exported a frozen graph of our detection model for TensorFlow Lite, we still need run it through the TensorFlow Lite Optimizing Converter (TOCO) before it will work with the TensorFlow Lite interpreter. TOCO converts models into an optimized FlatBuffer format that allows them to run efficiently on TensorFlow Lite. We also need to create a new label map before running the model.

#### Step 3a. Create optimized TensorFlow Lite model
First, we’ll run the model through TOCO to create an optimzed TensorFLow Lite model. The TOCO tool lives deep in the C:\tensorflow-build directory, and it will be run from the “tensorflow-build” Anaconda virtual environment that we created and used during Step 2. Meanwhile, the model we trained in Step 1 lives inside the C:\tensorflow1\models\research\object_detection\TFLite_model directory. We’ll create an environment variable called OUTPUT_DIR that points at the correct model directory to make it easier to enter the TOCO command.

If you don't already have an Anaconda Prompt window open with the "tensorflow-build" environment active and working in C:\tensorflow-build, open a new Anaconda Prompt window and issue:

```
activate tensorflow-build
cd C:\tensorflow-build
```

Create the OUTPUT_DIR environment variable by issuing:

```
set OUTPUT_DIR=C:\\tensorflow1\models\research\object_detection\TFLite_model
```

Next, use Bazel to run the model through the TOCO tool by issuing this command:

```
bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=%OUTPUT_DIR%/tflite_graph.pb --output_file=%OUTPUT_DIR%/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops 
```

Note: If you are using a floating, non-quantized SSD model (e.g. the ssdlite_mobilenet_v2_coco model rather than the ssd_mobilenet_v2_quantized_coco model), the Bazel TOCO command must be modified slightly:

```
bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=$OUTPUT_DIR/tflite_graph.pb --output_file=$OUTPUT_DIR/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --inference_type=FLOAT --allow_custom_ops 
```

If you are using Linux, make sure to use the commands given in the [official TensorFlow instructions here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md). I removed the ' characters from the command, because for some reason they cause errors on Windows!

After the command finishes running, you should see a file called detect.tflite in the \object_detection\TFLite_model directory. This is the model that can be used with TensorFlow Lite!

#### Step 3b. Create new label map
For some reason, TensorFlow Lite uses a different label map format than classic TensorFlow. The classic TensorFlow label map format looks like this (you can see an example in the \object_detection\data\mscoco_label_map.pbtxt file): 

```
item { 
  name: "/m/01g317" 
  id: 1 
  display_name: "person" 
} 
item { 
  name: "/m/0199g" 
  id: 2 
  display_name: "bicycle" 
} 
item { 
  name: "/m/0k4j" 
  id: 3 
  display_name: "car" 
} 
item { 
  name: "/m/04_sv" 
  id: 4 
  display_name: "motorcycle" 
} 
And so on...
```

However, the label map provided with the [example TensorFlow Lite object detection model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) looks like this:

```
person 
bicycle 
car 
motorcycle 
And so on...
```
 
Basically, rather than explicitly stating the name and ID number for each class like the classic TensorFlow label map format does, the TensorFlow Lite format just lists each class. To stay consistent with the example provided by Google, I’m going to stick with the TensorFlow Lite label map format for this guide.

Thus, we need to create a new label map that matches the TensorFlow Lite style. Open a text editor and list each class in order of their class number. Then, save the file as “labelmap.txt” in the TFLite_model folder. As an example, here's what the labelmap.txt file for my bird/squirrel/raccoon detector looks like:

<p align="center">
   <img src="doc/labelmap_example.png">
</p>
 
Now we’re ready to run the model!

#### Step 3c. Run the TensorFlow Lite model!
I wrote three Python scripts to run the TensorFlow Lite object detection model on an image, video, or webcam feed: TFLite_detection_image.py, TFLite_detection_video.py, and [TFLite_detection_wecam.py](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_webcam.py). The scripts are based off the label_image.py example given in the [TensorFlow Lite examples GitHub repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py).

We’ll download the Python scripts directly from this repository. First, install wget for Anaconda by issuing:

```
conda install -c menpo wget
```

Once it's installed, download the scripts by issuing:

```
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_image.py --no-check-certificate
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_video.py --no-check-certificate
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_webcam.py --no-check-certificate
```

The following instructions show how to run the webcam, video, and image scripts. These instructions assume your .tflite model file and labelmap.txt file are in the “TFLite_model” folder in your \object_detection directory as per the instructions given in this guide.

If you’d like try using the sample TFLite object detection model provided by Google, simply download it [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) and unzip it into the \object_detection folder. Then, use `--modeldir=coco_ssd_mobilenet_v1_1.0_quant_2018_06_29` rather than `--modeldir=TFLite_model` when running the script. 

For more information on options that can be used while running the scripts, use the `-h` option when calling the script. For example:

```
python TFLite_detection_image.py -h
```

##### Webcam
Make sure you have a USB webcam plugged into your computer. If you’re on a laptop with a built-in camera, you don’t need to plug in a USB webcam. 

From the \object_detection directory, issue: 

```
python TFLite_detection_webcam.py --modeldir=TFLite_model 
```

After a few moments of initializing, a window will appear showing the webcam feed. Detected objects will have bounding boxes and labels displayed on them in real time.

##### Video stream
To run the script to detect images in a video stream (e.g. a remote security camera), issue: 

```
python TFLite_detection_stream.py --modeldir=TFLite_model --streamurl="http://ipaddress:port/stream/video.mjpeg" 
```

After a few moments of initializing, a window will appear showing the video stream. Detected objects will have bounding boxes and labels displayed on them in real time.

Make sure to update the URL parameter to the one that's being used by your security camera. It has to include authentication information in case the stream is secured.

If the bounding boxes are not matching the detected objects, probably the stream resolution wasn't detected. In this case you can set it explicitly by using the `--resolution` parameter:

```
python TFLite_detection_stream.py --modeldir=TFLite_model --streamurl="http://ipaddress:port/stream/video.mjpeg" --resolution=1920x1080
```

##### Video
To run the video detection script, issue:

```
python TFLite_detection_image.py --modeldir=TFLite_model
```

A window will appear showing consecutive frames from the video, with each object in the frame labeled. Press 'q' to close the window and end the script. By default, the video detection script will open a video named 'test.mp4'. To open a specific video file, use the `--video` option:

```
python TFLite_detection_image.py --modeldir=TFLite_model --video='birdy.mp4'
```

Note: Video detection will run at a slower FPS than realtime webcam detection. This is mainly because loading a frame from a video file requires more processor I/O than receiving a frame from a webcam.

##### Image
To run the image detection script, issue:

```
python TFLite_detection_image.py --modeldir=TFLite_model
```

The image will appear with all objects labeled. Press 'q' to close the image and end the script. By default, the image detection script will open an image named 'test1.jpg'. To open a specific image file, use the `--image` option:

```
python TFLite_detection_image.py --modeldir=TFLite_model --image=squirrel.jpg
```

It can also open an entire folder full of images and perform detection on each image. There can only be images files in the folder, or errors will occur. To specify which folder has images to perform detection on, use the `--imagedir` option:

```
python TFLite_detection_image.py --modeldir=TFLite_model --imagedir=squirrels
```

Press any key (other than 'q') to advance to the next image. Do not use both the --image option and the --imagedir option when running the script, or it will throw an error.

<p align="center">
   <img src="doc/squirrels!!.png">
</p>

If you encounter errors while running these scripts, please check the [FAQ section](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#frequently-asked-questions-and-common-errors) of this guide. It has a list of common errors and their solutions. If you can successfully run the script, but your object isn’t detected, it is most likely because your model isn’t accurate enough. The FAQ has further discussion on how to resolve this.

### Next Steps
This concludes Part 1 of my TensorFlow Lite guide! You now have a trained TensorFlow Lite model and the scripts needed to run it on a PC.

But who cares about running it on a PC? The whole reason we’re using TensorFlow Lite is so we can run our models on lightweight devices that are more portable and less power-hungry than a PC!  The next two parts of my guide show how to run this TFLite model on a Raspberry Pi or an Android Device. 
