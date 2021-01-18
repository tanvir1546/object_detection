# object_detection
##Requirements
(Visual Studio 2019 with C++ Build Tools is required.https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16)
( Visual C++ 2015 build is required https://go.microsoft.com/fwlink/?LinkId=691126)
CUDA enabled devices https://developer.nvidia.com/cuda-gpus
Install NVIDIA DRIVER https://www.nvidia.com/Download/index.aspx
Install Install CUDA TOOLKIT v11.1 https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_456.81_win10.exe
Download cuDNN https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.1_20201106/cudnn-11.1-windows-x64-v8.0.5.39.zip
Download Anaconda https://www.anaconda.com/products/individual
Create virtual environment.  conda create -n tensorflow pip python=3.8
activate virtual Environment   conda activate tensorflow
install tensorflow gpu 
conda activate tensorflow
python
>>> import tensorflow as tf
>>> print(tf.__version__)
mkdir TensorFlow
cd C:\TensorFlow
git clone https://github.com/tensorflow/models.git
conda install -c anaconda protobuf
cd models\research
protoc object_detection\protos\*.proto --python_out=.
Open Anaconda promt
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
install pycocotools using following command
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI  ( Visual C++ 2015 build is required https://go.microsoft.com/fwlink/?LinkId=691126)
cd C:\TensorFlow\models\research
copy object_detection\packages\tf2\setup.py .
python -m pip install .
check if everything is working perfectly.
python object_detection\builders\model_builder_tf2_test.py
cd C:\TensorFlow\scripts\preprocessing
python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\train -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\train.record
python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\test -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\test.record
place the downloaded model in pre-trained model
copy and edit pipeline.config to models/my_ssd_mobilenet_v2_fpnlite
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
cd C:\TensorFlow\workspace\training_demo
python model_main_tf2.py --model_dir=models\my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models\my_ssd_mobilenet_v2_fpnlite\pipeline.config

monitoring training with tensorboard
conda activate tensorflow
cd C:\TensorFlow\workspace\training_demo
tensorboard --logdir=models\my_ssd_mobilenet_v2_fpnlite

cd C:\TensorFlow\workspace\training_demo


python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir .\models\my_ssd_mobilenet_v2_fpnlite\ --output_directory .\exported-models\my_mobilenet_model



Evaluating the Model

cd C:\TensorFlow\workspace\training_demo
python model_main_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --model_dir models\my_ssd_mobilenet_v2_fpnlite --checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --alsologtostderr
