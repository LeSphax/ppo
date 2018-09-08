import tensorflow as tf

def prepare_image(image):
    scaled_images = tf.cast(image, tf.float32) / 255.
    reshaped_layer = tf.reshape(scaled_images, tf.shape(image))
    return reshaped_layer