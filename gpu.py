import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(gpu_devices)
#tf.config.experimental.set_memory_growth(gpu_devices[0], True)
