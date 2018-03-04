r"""An example of how to use Tensorboard .

In this case, this script will:
  * Fine-tune a pre-trained MobileNet on Cifar-10 dataset
  * Summary the training process on Tensorboard
  * Visualize t-SNE
"""
import tensorflow as tf

_NUM_CLASSES = 10
_SHUFFLE_BUFFER = 100
_HEIGHT, _WIDTH, _DEPTH = 128, 128, 3

############################################
# Data processing
############################################
def preprocess(image, label, is_training):
  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT, _WIDTH)

  label = tf.one_hot(label[0], _NUM_CLASSES)
  if is_training:  # perform augmentation
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
    image = tf.image.random_flip_left_right(image)
  return image, label


def main():
  # Load cifar-10 Dataset
  cifar10 = tf.keras.datasets.cifar10.load_data()

  # Create an Estimator for training/evaluation
  classifier = get_estimator(model_function=cifar10_mobilenet_fn)

  print('Starting a training cycle.')
  images, labels = cifar10[0]

  classifier.train(
      input_fn=lambda: input_fn(
          is_training=True,
          num_epochs=10,
          batch_size=32,
          preprocess_fn=preprocess,
          shuffle_buffer=_SHUFFLE_BUFFER, num_parallel_calls=5,
          dataset=tf.data.Dataset.from_tensor_slices((images, labels))),
      hooks=[tf.train.LoggingTensorHook(
                tensors={'learning_rate': 'learning_rate',
                         'cross_entropy': 'cross_entropy',
                         'train_accuracy': 'train_accuracy'},
                every_n_iter=100)])

  print('Starting to evaluate.')
  test_images, test_labels = cifar10[0]

  eval_results = classifier.evaluate(
    input_fn=lambda: input_fn(
        is_training=False,
        num_epochs=1,
        batch_size=32,
        preprocess_fn=preprocess,
        dataset=tf.data.Dataset.from_tensor_slices((test_images, test_labels)),
        shuffle_buffer=_SHUFFLE_BUFFER, num_parallel_calls=5),)

  print(eval_results)


def get_estimator(model_function):
  session_config = tf.ConfigProto(inter_op_parallelism_threads=4,
                                  intra_op_parallelism_threads=4)
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e-9,
                                                session_config=session_config)
  classifier = tf.estimator.Estimator(
      model_fn=model_function,
      config=run_config,
      params={'multi_gpu': False})

  return classifier


def cifar10_mobilenet_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  weight_decay = 2e-4
  momentum = 0.9
  initial_learning_rate = 0.1
  learning_rate = initial_learning_rate * 0.001

  # Create a tensor named learning_rate for logging purposes
  tf.identity(learning_rate, name='learning_rate')
  tf.summary.scalar('learning_rate', learning_rate)

  input_layer = tf.keras.layers.Input(
      shape=(_HEIGHT, _WIDTH, _DEPTH),
      name='image')

  # Load pre-trained model
  model = tf.keras.applications.MobileNet(
      input_tensor=input_layer,
      include_top=True, weights='imagenet')
  # Remove the last two layer (Conv2D, Reshape) for fine-tuning on CIFAR-10
  model = tf.keras.Model(model.inputs, model.layers[-4].output)

  return model_fn(features, labels, mode, model,
                  optimizer=tf.train.MomentumOptimizer(learning_rate, momentum),
                  weight_decay=weight_decay)


def model_fn(features, labels, mode, model, weight_decay, optimizer,
             multi_gpu=False):
  """Construct `model_fn` for estimator

  Returns:
    EstimatorSpec
  """
  tf.summary.image('images', features, max_outputs=6)
  model = model(features)
  logits = tf.keras.layers.Conv2D(_NUM_CLASSES, (1, 1),  padding='same',
                                  activation='softmax')(model)
  logits = tf.keras.layers.Reshape((_NUM_CLASSES, ), name='logits')(logits)

  predictions = {
      'classes': tf.argmax(logits),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # L2 regularization
  loss = cross_entropy + weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'bn' not in v.name])   # Exclude batch_norm

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    # Batch norm requires update ops to be added
    # as a dependency to the train op.

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  # Create a tensor named `train_accuracy` for logging
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={'accuracy': accuracy})


def input_fn(is_training, dataset, preprocess_fn,
             batch_size, shuffle_buffer, num_epochs,
             num_parallel_calls=1, multi_gpu=False):
  """

  Args:
  Returns:

  """
  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  dataset = dataset.map(lambda img, label: preprocess_fn(img, label, is_training),
                        num_parallel_calls=num_parallel_calls)

  dataset = dataset.batch(batch_size)
  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path.
  dataset = dataset.prefetch(1)

  return dataset


if __name__ == '__main__':
    main()
