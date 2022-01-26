# Multi-modal-Registration

Stage1:

CycleGAN-Tensorflow for image translation:

Environment:
TensorFlow 1.0.0
Python 3.6.0

Data preparing

First download the dataset e.g. HE2Ki67

$ bash download_dataset.sh HE2Ki67

Write the dataset to tfrecords

$ python build_data.py

Check $ python3 build_data.py --help for more details.

Training

$ python train.py

To change some default settings, pass those to the command line, such as:

$ python train.py  \
    --X=data/tfrecords/HE.tfrecords \
    --Y=data/tfrecords/Ki67.tfrecords
    
Here is the list of arguments:

