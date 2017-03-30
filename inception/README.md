## Directory for train,eval and data scripts

To build your train, eval and test scripts, modify the provided flowers_data.py, flowers_eval.py and flowers_train.py scripts.
PS: For the purpose of this tutorial I will refer to your test usecase as DR (eg. Flowers) therefore I will rename the above scripts as DR_data.py, DR_eval.py and DR_train.py

Example for change in flowers_data.py which is saved as DR_data.py

```python
class FlowersData(Dataset):
  """Flowers data set."""

  def __init__(self, subset):
    super(FlowersData, self).__init__('Flowers', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 5          ### Change the number of classes to your number of classes 

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return 3170     ### Change the number of training data points
    if self.subset == 'validation':
      return 500      ### Change the number of validation data points
```

Example for change in flowers_train.py which is saved as DR_data.py

```python 
from inception.flowers_data import FlowersData  ### Import your data function

FLAGS = tf.app.flags.FLAGS


def main(_):
  dataset = FlowersData(subset=FLAGS.subset)  ### Call your data function
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  inception_train.train(dataset)


if __name__ == '__main__':
  tf.app.run()
```

Example for change in flowers_eval.py which is saved as DR_eval.py

```python
from inception.flowers_data import FlowersData  ### Import your data function

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
  dataset = FlowersData(subset=FLAGS.subset)   ### Call your data function
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  inception_eval.evaluate(dataset)


if __name__ == '__main__':
  tf.app.run()
  ```
  
  
