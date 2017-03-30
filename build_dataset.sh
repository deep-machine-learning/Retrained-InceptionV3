# location to where to save the TFRecord data.
OUTPUT_DIRECTORY=/media/analytics/data/DiabeticRetinopathy/data/model-ready-data

# build the preprocessing script.
bazel build inception/build_image_data

# convert the data.
bazel-bin/inception/build_image_data \
  --train_directory="/media/analytics/data/DiabeticRetinopathy/data/TRAIN_DIR" \
  --validation_directory="/media/analytics/data/DiabeticRetinopathy/data/VAL_DIR" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="/media/analytics/data/DiabeticRetinopathy/labels.txt" \
  --train_shards=24\
  --validation_shards=8 \
  --num_threads=8
