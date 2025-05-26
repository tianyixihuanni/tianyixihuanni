import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
# import tensorflow as tf
#
# # 打印可用的设备列表
# print("Available devices:")
# for device in tf.config.experimental.list_physical_devices():
#     print(device)
#
# # 检查是否有可用的GPU
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))