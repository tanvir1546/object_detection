# Installs Prerequisites for OpenCV
pip3 install cython
pip3 install wheel
sudo apt-get install libhdf5-dev libhdf5-serial-dev
sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt-get install libatlas-base-dev
python3 -m pip install --upgrade pip
pip3 install numpy
python3 -m pip install --upgrade Pillow
pip3 install matplotlib
pip3 install pandas

# Installs OpenCV pip package
pip3 install 


pip install opencv-python

# Install the tflite_runtime
pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp36-cp36m-linux_aarch64.whl
echo Prerequisites Installed Successfully
