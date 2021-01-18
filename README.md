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

#### Step 2a. Install MSYS2
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


#

#### Step 2c. Update Anaconda and create tensorflow-build environment
Now that the Visual Studio tools are installed and your PC is freshly restarted, open a new Anaconda Prompt window. First, update Anaconda to make sure its package list is up to date. In the Anaconda Prompt window, issue these two commands: 

```
conda update -n base -c defaults conda
conda update --all
```

The update process may take up to an hour, depending on how it's been since you installed or updated Anaconda. Next, create a new Anaconda virtual environment called “tensorflow-build”. We’ll work in this environment for the rest of the build process. Create and activate the environment by issuing:

```
conda create -n tensorflow-build pip python=3.8
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
pip install keras_applications --no-deps
pip install keras_preprocessing --no-deps
```

```
conda install -c conda-forge bazel=3.1.0
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
git checkout r2.4
```

Next, we’ll configure the TensorFlow build using the configure.py script. From the C:\tensorflow-build\tensorflow directory, issue:

```
python ./configure.py
```

#### Step 2e. Build TensorFlow package


```
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package 
```


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






# Part 2 - How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi (with Optional Coral USB Accelerator)


## Section 1 - How to Set Up and Run TensorFlow Lite Object Detection Models on the Raspberry Pi

S

### Step 1a. Update the Raspberry Pi
First, the Raspberry Pi needs to be fully updated. Open a terminal and issue:
```
sudo apt-get update
sudo apt-get dist-upgrade
```
Depending on how long it’s been since you’ve updated your Pi, the update could take anywhere between a minute and an hour. 

While we're at it, let's make sure the camera interface is enabled in the Raspberry Pi Configuration menu. Click the Pi icon in the top left corner of the screen, select Preferences -> Raspberry Pi Configuration, and go to the Interfaces tab and verify Camera is set to Enabled. If it isn't, enable it now, and reboot the Raspberry Pi.

<p align="center">
  <img src="/doc/camera_enabled.png">
</p>

### Step 1b. Download this repository and create virtual environment

Next, clone this GitHub repository by issuing the following command. The repository contains the scripts we'll use to run TensorFlow Lite, as well as a shell script that will make installing everything easier. Issue:

```
git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git
```

This downloads everything into a folder called TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi. That's a little long to work with, so rename the folder to "tflite1" and then cd into it:

```
mv TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi tflite1
cd tflite1
```

We'll work in this /home/pi/tflite1 directory for the rest of the guide. Next up is to create a virtual environment called "tflite1-env".

I'm using a virtual environment for this guide because it prevents any conflicts between versions of package libraries that may already be installed on your Pi. Keeping TensorFlow installed in its own environment allows us to avoid version conflicts. For example, if you've already installed TensorFlow v1.8 on the Pi using my [other guide](https://www.youtube.com/watch?v=npZ-8Nj1YwY), you can leave that installation as-is without having to worry about overriding it.

Install virtualenv by issuing:

```
sudo pip3 install virtualenv
```

Then, create the "tflite1-env" virtual environment by issuing:

```
python3 -m venv tflite1-env
```

This will create a folder called tflite1-env inside the tflite1 directory. The tflite1-env folder will hold all the package libraries for this environment. Next, activate the environment by issuing:

```
source tflite1-env/bin/activate
```

**You'll need to issue the `source tflite1-env/bin/activate` command from inside the /home/pi/tflite1 directory to reactivate the environment every time you open a new terminal window. You can tell when the environment is active by checking if (tflite1-env) appears before the path in your command prompt, as shown in the screenshot below.**

At this point, here's what your tflite1 directory should look like if you issue `ls`.

<p align="center">
  <img src="/doc/tflite1_folder.png">
</p>

If your directory looks good, it's time to move on to Step 1c!

### Step 1c. Install TensorFlow Lite dependencies and OpenCV
Next, we'll install TensorFlow, OpenCV, and all the dependencies needed for both packages. OpenCV is not needed to run TensorFlow Lite, but the object detection scripts in this repository use it to grab images and draw detection results on them.

To make things easier, I wrote a shell script that will automatically download and install all the packages and dependencies. Run it by issuing:

```
bash get_pi_requirements.sh
```

This downloads about 400MB worth of installation files, so it will take a while. Go grab a cup of coffee while it's working! If you'd like to see everything that gets installed, simply open get_pi_dependencies.sh to view the list of packages.

**NOTE: If you get an error while running the `bash get_pi_requirements.sh` command, it's likely because your internet connection timed out, or because the downloaded package data was corrupted. If you get an error, try re-running the command a few more times.**

**ANOTHER NOTE: The shell script automatically installs the latest version of TensorFlow. If you'd like to install a specific version, issue `pip3 install tensorflow==X.XX` (where X.XX is replaced with the version you want to install) after running the script. This will override the existing installation with the specified version.**

That was easy! On to the next step.

### Step 1d. Set up TensorFlow Lite detection model
Next, we'll set up the detection model that will be used with TensorFlow Lite. This guide shows how to either download a sample TFLite model provided by Google, or how to use a model that you've trained yourself by following [Part 1 of my TensorFlow Lite tutorial series](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#part-1---how-to-train-convert-and-run-custom-tensorflow-lite-object-detection-models-on-windows-10).

A detection model has two files associated with it: a detect.tflite file (which is the model itself) and a labelmap.txt file (which provides a labelmap for the model). My preferred way to organize the model files is to create a folder (such as "BirdSquirrelRaccoon_TFLite_model") and keep both the detect.tflite and labelmap.txt in that folder. This is also how Google's downloadable sample TFLite model is organized.

#### Option 1. Using Google's sample TFLite model
Google provides a sample quantized SSDLite-MobileNet-v2 object detection model which is trained off the MSCOCO dataset and converted to run on TensorFlow Lite. It can detect and identify 80 different common objects, such as people, cars, cups, etc.

Download the sample model (which can be found on [the Object Detection page of the official TensorFlow website](https://www.tensorflow.org/lite/models/object_detection/overview)) by issuing:

```
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

Unzip it to a folder called "Sample_TFLite_model" by issuing (this command automatically creates the folder):

```
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model
```

Okay, the sample model is all ready to go! 

#### Option 2: Using your own custom-trained model
You can also use a custom object detection model by moving the model folder into the /home/pi/tflite directory. If you followed [Part 1 of my TensorFlow Lite guide](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#part-1---how-to-train-convert-and-run-custom-tensorflow-lite-object-detection-models-on-windows-10) to train and convert a TFLite model on your PC, you should have a folder named "TFLite_model" with a detect.tflite and labelmap.txt file. (It will also have a tflite_graph.pb and tflite_graph.pbtxt file, which are not needed by TensorFlow Lite but can be left in the folder.) 

You can simply copy that folder to a USB drive, insert the USB drive in your Raspberry Pi, and move the folder into the /home/pi/tflite1 directory. (Or you can email it to yourself, or put it on Google Drive, or do whatever your preferred method of file transfer is.) Here's an example of what my "BirdSquirrelRaccoon_TFLite_model" folder looks like in my /home/pi/tflite1 directory: 

<p align="center">
  <img src="/doc/BSR_directory1.png">
</p>

Now your custom model is ready to go!

### Step 1e. Run the TensorFlow Lite model!
It's time to see the TFLite object detection model in action! First, free up memory and processing power by closing any applications you aren't using. Also, make sure you have your webcam or Picamera plugged in.

Run the real-time webcam detection script by issuing the following command from inside the /home/pi/tflite1 directory. (Before running the command, make sure the tflite1-env environment is active by checking that (tflite1-env) appears in front of the command prompt.) **The TFLite_detection_webcam.py script will work with either a Picamera or a USB webcam.**

```
python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model
```

If your model folder has a different name than "Sample_TFLite_model", use that name instead. For example, I would use `--modeldir=BirdSquirrelRaccoon_TFLite_model` to run my custom bird, squirrel, and raccoon detection model.

After a few moments of initializing, a window will appear showing the webcam feed. Detected objects will have bounding boxes and labels displayed on them in real time.

Part 3 of my TensorFlow Lite training guide gives [instructions](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#video) for using the TFLite_detection_image.py and TFLite_detection_video.py scripts. Make sure to use `python3` rather than `python` when running the scripts.

## Section 2 - Run Edge TPU Object Detection Models on the Raspberry Pi Using the Coral USB Accelerator

[![Link to Section 2 YouTube video!](https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/doc/YouTube_video2.png)](https://www.youtube.com/watch?v=qJMwNHQNOVU)

The [Coral USB Accelerator](https://coral.withgoogle.com/products/accelerator/) is a USB hardware accessory for speeding up TensorFlow models. You can buy one [here (Amazon Associate link)](https://amzn.to/2BuG1Tv). 

The USB Accelerator uses the Edge TPU (tensor processing unit), which is an ASIC (application-specific integrated circuit) chip specially designed with highly parallelized ALUs (arithmetic logic units). While GPUs (graphics processing units) also have many parallelized ALUs, the TPU has one key difference: the ALUs are directly connected to eachother. The output of one ALU can be directly passed to the input of the next ALU without having to be stored and retrieved from a memory buffer. The extreme paralellization and removal of the memory bottleneck means the TPU can perform up to 4 trillion arithmetic operations per second! This is perfect for running deep neural networks, which require millions of multiply-accumulate operations to generate outputs from a single batch of input data. 

<p align="center">
  <img src="/doc/Coral_and_EdgeTPU2.png">
</p>

My Master's degree was in ASIC design, so the Edge TPU is very interesting to me! If you're a computer architecture nerd like me and want to learn more about the Edge TPU, [here is a great article that explains how it works](https://cloud.google.com/blog/products/ai-machine-learning/what-makes-tpus-fine-tuned-for-deep-learning).

It makes object detection models run WAY faster, and it's easy to set up. These are the steps we'll go through to set up the Coral USB Accelerator:

- 2a. Install libedgetpu library
- 2b. Set up Edge TPU detection model
- 2c. Run super-speed detection!

This section of the guide assumes you have already completed [Section 1](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md#section-1---how-to-set-up-and-run-tensorflow-lite-object-detection-models-on-the-raspberry-pi) for setting up TFLite object detection on the Pi. If you haven't done that portion, scroll back up and work through it first.

### Step 2a. Install libedgetpu library
First, we'll download and install the Edge TPU runtime, which is the library needed to interface with the USB Acccelerator. These instructions follow the [USB Accelerator setup guide](https://coral.withgoogle.com/docs/accelerator/get-started/) from official Coral website.

Open a command terminal and move into the /home/pi/tflite1 directory and activate the tflite1-env virtual environment by issuing:

```
cd /home/pi/tflite1
source tflite1-env/bin/activate
```

Add the Coral package repository to your apt-get distribution list by issuing the following commands:

```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
```

Install the libedgetpu library by issuing:

```
sudo apt-get install libedgetpu1-std
```

You can also install the libedgetpu1-max library, which runs the USB Accelerator at an overclocked frequency, allowing it to achieve even faster framerates. However, it also causes the USB Accelerator to get hotter. Here are the framerates I get when running TFLite_detection_webcam.py with 1280x720 resolution for each option with a Raspberry Pi 4 4GB model:

* libedgetpu1-std: 22.6 FPS
* libedgetpu1-max: 26.1 FPS

I didn't measure the temperature of the USB Accelerator, but it does get a little hotter to the touch with the libedgetpu1-max version. However, it didn't seem hot enough to be unsafe or harmful to the electronics.

If you want to use the libedgetpu-max library, install it by using `sudo apt-get install libedgetpu1-max`. (You can't have both the -std and the -max libraries installed. If you install the -max library, the -std library will automatically be uninstalled.)

Alright! Now that the libedgetpu runtime is installed, it's time to set up an Edge TPU detection model to use it with.

### Step 2b. Set up Edge TPU detection model
Edge TPU models are TensorFlow Lite models that have been compiled specifically to run on Edge TPU devices like the Coral USB Accelerator. They reside in a .tflite file and are used the same way as a regular TF Lite model. My preferred method is to keep the Edge TPU file in the same model folder as the TFLite model it was compiled from, and name it as "edgetpu.tflite".

I'll show two options for setting up an Edge TPU model: using the sample model from Google, or using a custom model you compiled yourself.

#### Option 1. Using Google's sample EdgeTPU model
Google provides a sample Edge TPU model that is compiled from the quantized SSDLite-MobileNet-v2 we used in [Step 1e](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md#step-1e-set-up-tensorflow-lite-detection-model). Download it and move it into the Sample_TFLite_model folder (while simultaneously renaming it to "edgetpu.tflite") by issuing these commands:

```
wget https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite

mv mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite Sample_TFLite_model/edgetpu.tflite
```

Now the sample Edge TPU model is all ready to go. It will use the same labelmap.txt file as the TFLite model, which should already be located in the Sample_TFLite_model folder.

#### Option 2. Using your own custom EdgeTPU model
If you trained a custom TFLite detection model, you can compile it for use with the Edge TPU. Unfortunately, the edgetpu-compiler package doesn't work on the Raspberry Pi: you need a Linux PC to use it on. Section 3 of this guide will give a couple options for compiling your own model if you don't have a Linux box. While I'm working on writing it, [here are the official instructions that show how to compile an Edge TPU model from a TFLite model](https://coral.withgoogle.com/docs/edgetpu/compiler/).

Assuming you've been able to compile your TFLite model into an EdgeTPU model, you can simply copy the .tflite file onto a USB and transfer it to the model folder on your Raspberry Pi. For my "BirdSquirrelRaccoon_TFLite_model" example from [Step 1e](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md#step-1e-set-up-tensorflow-lite-detection-model), I can compile my "BirdSquirrelRaccoon_TFLite_model" on a Linux PC, put the resulting edgetpu.tflite file on a USB, transfer the USB to my Pi, and move the edgetpu.tflite file into the /home/pi/tflite1/BirdSquirrelRaccoon_TFLite_model folder. It will use the same labelmap.txt file that already exists in the folder to get its labels.

Once the edgetpu.tflite file has been moved into the model folder, it's ready to go!

### Step 2c. Run detection with Edge TPU!

Now that everything is set up, it's time to test out the Coral's ultra-fast detection speed! Make sure to free up memory and processing power by closing any programs you aren't using. Make sure you have a webcam plugged in.

Plug in your Coral USB Accelerator into one of the USB ports on the Raspberry Pi. If you're using a Pi 4, make sure to plug it in to one of the blue USB 3.0 ports.

*Insert picture of Coral USB Accelerator plugged into Raspberry Pi here!*

Make sure the tflite1-env environment is activate by checking that (tflite1-env) appears in front of the command prompt in your terminal. Then, run the real-time webcam detection script with the --edgetpu argument:

```
python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model --edgetpu
```

The `--edgetpu` argument tells the script to use the Coral USB Accelerator and the EdgeTPU-compiled .tflite file. If your model folder has a different name than "Sample_TFLite_model", use that name instead.

After a brief initialization period, a window will appear showing the webcam feed with detections drawn on each from. The detection will run SIGNIFICANTLY faster with the Coral USB Accelerator.

If you'd like to run the video or image detection scripts with the Accelerator, use these commands:

```
python3 TFLite_detection_video.py --modeldir=Sample_TFLite_model --edgetpu
python3 TFLite_detection_image.py --modeldir=Sample_TFLite_model --edgetpu
```

Have fun with the blazing detection speeds of the Coral USB Accelerator!

## Section 3 - Compile Custom Edge TPU Object Detection Models

To use a custom model on the Coral USB Accelerator, you have to run it through Coral's [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/) tool. Unfortunately, the compiler only works on Linux operating systems, and only on certain CPU architectures. 

The easiest way to compile the Edge TPU model is to use a Google Colab session. I created a Colab page specifically for compiling Edge TPU models. Please click the link below and follow the instructions in the Colab notebook.

https://colab.research.google.com/drive/1o6cNNNgGhoT7_DR4jhpMKpq3mZZ6Of4N?usp=sharing

