# Retraining InceptionV3 Model using a new dataset (Transfer Learning)

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

## 
