# Retraining InceptionV3 Model using a new dataset (Transfer Learning)

## Downloading imagenet trained InceptionV3 model

```
# location of where to place the Inception v3 model
DATA_DIR=$HOME/inception-v3-model
mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

# download the Inception v3 model
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
tar xzf inception-v3-2016-03-01.tar.gz

# this will create a directory called inception-v3 which contains the following files.
> ls inception-v3
README.txt
checkpoint
model.ckpt-157585
```

## Data Processing

It is important to pre-process your data in order to make the process of retraining inceptionV3 easier. Configure your data folder the following way:

The image data set is expected to reside in JPEG files located in the
following directory structure.

```
  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128
```

Now, run *./Retrained-InceptionV3/Inception/data/build_image_data.py* to convert your image data to TFRecords format as expected by the inceptionV3 model. 

## Script preparation 

Follow the instructions given in the **README** file at *./Retrained-InceptionV3/Inception* folder

## Training 

**DR_run_train.sh** file has the required instructions to run the training on your dataset once the above steps are finished. The script has the following:

```
DIR=${HOME_DIR}

# Build the model. Note that we need to make sure the TensorFlow is ready to
# use before this as this command will not build TensorFlow.
bazel build inception/DR_train

# Path to the downloaded Inception-v3 model.
MODEL_PATH="${DIR}/inception-v3/model.ckpt-157585"

# Directory where the flowers data resides.
DR_DATA_DIR="${DIR}/data/model-ready-data"

# Directory where to save the checkpoint and events files.
TRAIN_DIR="${DIR}/DR_chpk"

# Run the fine-tuning on the flowers data set starting from the pre-trained
# Imagenet-v3 model.
bazel-bin/inception/DR_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${DR_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_fiactor=1 \
  --max_steps=10000 \
  --batch_size=64
  
 ```
 Fine tune the hyperparameters to your dataset and requirments. 
 
 ## Freeze graph from checkpoint model
 
 The output of training will save the meta graph of the model in the .meta file and weights in the checkpoint file saved in the DR_chpk directory. Unfortunately, these files cannot be directly used for prediction. We have to freeze the graph definition along with the weights in one .pb file. 
 
 This can be achieved by the **freeze_graph.py** file. The output of this file is frozen_model.pb which can be used for prediction. 
 
 ## Prediction
 
 To use your retrained image classification model to make prediction on unknown images, use **predict.py**. We can either output the human string output by the softmax layer or the pre-softmax pool layer giving us a 2048 signature of the image. By default the top-k predictions is set to 1. You can change this by setting the *num_top_predictions* system arg. Example:
 
 ```
 python predict.py --image_dir=${DIR_OF_TEST_IMAGES} --layer='softmax' # for softmax layer 
 python predict.py --image_dir=${DIR_OF_TEST_IMAGES} --layer='pool' # for pool layer
 ```
 
 The current inceptionV3 architecture takes batches of 64 images with size 299x299x3 as an input. If your test directory has fewer than 64 images (eg. 1), **predict.py** script will pad the remaining images with zeros. If your test directory has more than 64 images, **predict.py** script will batch the them into chunks of 64 images. 
 
 ## FAQ
 
 1. What version of Tensorflow does this work on?
 
 v1.0.1
 
 2. What version of Python does this work on?
 
 v2.7
 
 3. Does the training have GPU support?
 
 Yes, this is tested on Nvidia TitanX
 
 
 



