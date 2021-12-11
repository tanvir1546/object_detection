# Installs Prerequisites for OpenCV
pip3 install cython
pip3 install wheel
sudo apt-get install libhdf5-dev libhdf5-serial-dev
sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt-get install libatlas-base-dev
pip3 install numpy
python3 -m pip install --upgrade Pillow
#pip3 install matplotlib
pip3 install pandas
# Installs OpenCV pip package
pip install opencv-python
# Install the tflite_runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime
echo Prerequisites Installed Successfully
