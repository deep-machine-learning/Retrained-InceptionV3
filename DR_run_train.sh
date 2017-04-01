DIR=${HOME_DIR}

# Build the model. Note that we need to make sure the TensorFlow is ready to
# use before this as this command will not build TensorFlow.
#bazel build inception/DR_train

# Path to the downloaded Inception-v3 model.
MODEL_PATH="${DIR}/inception/inception-v3/model.ckpt-157585"

# Directory where the flowers data resides.
DR_DATA_DIR="${DIR}/data/model-ready-data"

# Directory where to save the checkpoint and events files.
TRAIN_DIR="${DIR}\DR_chpk"

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
