
# Object Detection in Windows 10 and implementation in Raspberry Pi 4 with EDGE TPU 
## Requirements
#### 1. Visual Studio 2019 with C++ Build Tools is required for building tensorflow v2.4.0 (optional: if u want to build from the source)
https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16
#### 2.  Visual C++ 2015 build is required for installing pycocotools 
LINK:
https://go.microsoft.com/fwlink/?LinkId=691126

#### List of CUDA enabled devices 
LINK: https://developer.nvidia.com/cuda-gpus
#### If you have CUDA enabled device download and install drivers and Cuda toolkit.
#### 3. Install NVIDIA DRIVER 
LINK: https://www.nvidia.com/Download/index.aspx
#### 4. Install Install CUDA TOOLKIT v11.1 
LINK: https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_456.81_win10.exe
#### 5. Download cuDNN 
LINK: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.1_20201106/cudnn-11.1-windows-x64-v8.0.5.39.zip
After downloading extract the file. Copy ***bin , include, lib*** to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1
#### Add CUDA to your path.
 Press **window+R** and run **control sysdm.cpl**. Go to **Advanced >Environment Variable**. Click **New...**  
```
Variable name: CUDA_PATH
Variable value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1
```
6. Download Anaconda https://www.anaconda.com/products/individual

## PART 1
### STEP(1.a) Create and activate virtual environment and install TensorFlow.
Open **Command Promt**
Press Windows+R type "cmd" and hit enter. 
In command promt type following commnad.
```
conda create -n tensorflow pip python=3.8

conda activate tensorflow
```
#### Install tensorflow-gpu
```
pip install tensorflow-gpu
```
#### If you do not have GPU
```
pip install tensorflow
```
#### Test your installation 
```
python
  >>> import tensorflow as tf
  >>> print(tf.__version__)
  20xx-xx-xx xx:xx:xx.xxxxxx: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
  >>> print(tf.__version__)
  2.4.0
  >>> exit()
```

### STEP(1.b) Download Model and required git files

```
cd C:\
mkdir TensorFlow
cd C:\TensorFlow
conda install -c anaconda git
git clone https://github.com/tensorflow/models.git
cd models\research
protoc object_detection\protos\*.proto --python_out=.
C:\TensorFlow
git clone https://github.com/tanvir1546/object-detection.git
```
Close **Command Prompt**.
### STEP(1.c) Open Anaconda prompt and Install pre-requisites:
```
conda activate tensorflow
```
```
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
**Install pycocotools using following command**(Visual C++ 2015 build is required)
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
#### Check if everything is working perfectly.
```
python object_detection\builders\model_builder_tf2_test.py
```
### Download the model and place the downloaded model in pre-trained model
**Download ssd mobilenet models from here**.
http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
Download and extract the file. Copy the file to **C:\TensorFlow\workspace\training_demo\pre-trained-models**
And then copy **pre-trained-models/pipeline_config** to models/my_ssd_mobilenet_v2_fpnlite and edit **pipeline.config**


You can find other models in here... 
~~https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md~~
 
```
Line 3. num_classes: 2 (type of object u r detecting. )
Line 135. batch_size: 5 (adjust this for your computer spec...for higher value u need more memory)
Line 165. fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0"
Line 171. fine_tune_checkpoint_type: "detection"
Line 175. label_map_path: "annotations/label_map.pbtxt"
Line 177. input_path: "annotations/train.record"
Line 185. label_map_path: "annotations/label_map.pbtxt"
Line 189. input_path: "annotations/test.record"
```

**Congratulations! Your System is ready for Training..... **
## STEP 2: Preparing Datasets
### STEP(2.a)Open another anaconda prompt for labelling the image.. if you finished this earlier skip this step
```
cd C:\TensorFlow
git clone https://github.com/tzutalin/labelImg.git
conda install pyqt=5
conda install -c anaconda lxml
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```
Tips: keyboard shortcuts: 'a' and 'd' for previous and next image. w for RectBox. If you are working with same class go to view and check "Single class mode" and "Auto save mode" 
Save format should be PascalVOC.

### STEP(2.a) Divide your Dataset and Export test.record and train.record

Divide your dataset into test and train. 
```
cd C:\TensorFlow\scripts\preprocessing
python partition_dataset.py -i C:/TensorFlow/workspace/training_demo/images/raw -o C:/TensorFlow/workspace/training_demo/images/ -x

```
Export test.record and train.record
```
cd C:\TensorFlow\scripts\preprocessing
python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\train -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\train.record
python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\test -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\test.record
```
## STEP 3: Training
### STEP (3.a)Start training and forget about this . Depending on your dataset, this might take upto 5 hours or more.
```
cd C:\TensorFlow\workspace\training_demo
python model_main_tf2.py --model_dir=models\my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models\my_ssd_mobilenet_v2_fpnlite\pipeline.config
```
### STEP (3.b) Monitoring training with tensorboard- open another anaconda promt(Skip this if you want)
```
conda activate tensorflow
cd C:\TensorFlow\workspace\training_demo
tensorboard --logdir=models\my_ssd_mobilenet_v2_fpnlite
```
###  STEP (3.c) Exporting the Inference Graph
```
cd C:\TensorFlow\workspace\training_demo

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir .\models\my_ssd_mobilenet_v2_fpnlite\ --output_directory .\exported-models\my_mobilenet_model
```
###  STEP (3.d) Evaluating the Model(optional)
```
cd C:\TensorFlow\workspace\training_demo
python model_main_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --model_dir models\my_ssd_mobilenet_v2_fpnlite --checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --alsologtostderr
```
## STEP 4: Using the model
copy annotations/label_map.pbtxt to models/my_ssd_mobilenet_v2_fpnlite/label_map.pbtxt
```
cd C:\TensorFlow\workspace\training_demo
python TF-image-od.py --image image_path/image.jpg
```

**Congratulations! You have completed the training and finished training .... **

## PART 2: CONVERTING TO TF_LITE MODEL FOR Raspberry Pi
###  STEP (1.a) Exporting the Model as tf_lite

```
cd C:\TensorFlow\workspace\training_demo
python export_tflite_graph_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --output_directory exported-models\my_tflite_model
```
#### Creating a New Environment and Installing TensorFlow
```
conda deactivate
conda create -n tflite pip python=3.7
conda activate tflite
pip install tf-nightly
python
Python 3.7.9 (default, Aug 31 2020, 17:10:11) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version)
2.5.0
```

#### Converting the Model to TensorFlow Lite
```
python convert-to-tflite.py
```
#### labels.txt
Preparing our Model for Use ```exported-models\my_tflite_model\saved_model``` as ```labels.txt```. 
Trick is your delete all the extra things in lebel_map.pbtxt jst write down the class name.

### STEP (2.a) - Run TensorFlow Lite Object Detection Models on the Raspberry Pi
#### Download this repository and create virtual environment

```
git clone https://github.com/tanvir1546/object_detection.git
cd object_detection
mv raspi tflite
cd tflite
sudo pip3 install virtualenv
```

Then, create the "tflite-env" virtual environment by issuing:

```
python3 -m venv tflite-env
```

This will create a folder called tflite1-env inside the tflite directory. The tflite-env folder will hold all the package libraries for this environment. Next, activate the environment by issuing:

```
source tflite-env/bin/activate
```
### Step (2.b) Install TensorFlow Lite dependencies and OpenCV

```
cd tflite
bash install-prerequisits.sh
```

### Step (2.c) Set up TensorFlow Lite detection model
copy model.tflite and labels.txt from laptop to raspberry pi and place it to model folder
Tree of your file should be something like this.
```
tflite
  |
  |--model
      |
      |--model.tflite
      |--labels.txt
```
### STEP (99999999999999999)
Have fun with your model. 

## PART 3: PREPARING MODEL FOR EDGE_TPU 
Edge TPU can make your model run faster. you may get upto 40 FPS or more using EDGE_TPU. Little more to do.. 
### STEP(1.a) Convert to int8 for EDGE_TPU files are in laptop
DOWNLOAD schema_py_generated.py from
https://drive.google.com/file/d/154wFehrb03Ck84A-5NACHsB4IGirJWcX/view?usp=sharing
and copy it to C:\Users\**YOUR_USER_NAME**\anaconda3\envs\tflite\Lib\site-packages\tensorflow\lite\python

```
activate tflite
python int8.py
```
you will get output named model_full_integer_quant.tflite in export_tflite_graph/saved_model directory. One last step to do. 
Open following google colab file and upload the generated model.
google colab EDGE_TPU compiler
https://colab.research.google.com/drive/1o6cNNNgGhoT7_DR4jhpMKpq3mZZ6Of4N?usp=sharing#scrollTo=WTboEAWuJ0ku

Now replace the files in model folder in Raspberry pi and use the new files. Hope it will work.
### Use JETSON Nano
NOTE: You might want to use JP4.4 JP4.5 gives "core dumped" error...

```
sudo pip3 install virtualenv
```

Then, create the "tflite-env" virtual environment by issuing:

```
python3 -m venv tflite-env
```

This will create a folder called tflite1-env inside the tflite1 directory. The tflite1-env folder will hold all the package libraries for this environment. Next, activate the environment by issuing:

```
source tflite-env/bin/activate
```
Everything is same except we need to install separate version of tf-runtime as it 64 bit operating system
```
cd object_detection/jetson_nano
bash install-prerequisites.sh
```
IF everything is OK your program should work. For webcam live detection there are some issues with opencv... still working on that .. 

##### ACKNOWLEGEMENT
Based ON: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
Special Thanks to: 
https://github.com/EdjeElectronics/
https://github.com/armaanpriyadarshan/
