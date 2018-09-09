import numpy as np
import tensorflow as tf
import time

path = None
previous_summary_time = None
previous_step = None

train_writer = None

images = {}
values = {}
placeholders = {}


def init(summary_path):
    global path
    path = summary_path


def add(variable_name, value):
    if variable_name not in values:
        values[variable_name] = [value]
    else:
        values[variable_name].append(value)


def add_image(variable_name, value):
    images[variable_name] = value


def create_summaries():
    scalars = list(values.keys())
    scalars.append("Steps/s")
    for var_name in scalars:
        placeholders[var_name] = tf.placeholder(tf.float32, ())
        tf.summary.scalar(var_name, placeholders[var_name])

    for var_name in images:
        placeholders[var_name] = tf.placeholder(tf.float32, np.shape(images[var_name]))
        tf.summary.image(var_name, placeholders[var_name])

    global merged
    merged = tf.summary.merge_all()
    global sess
    sess = tf.get_default_session()
    global train_writer
    train_writer = tf.summary.FileWriter(path)


def save(current_step):
    if train_writer is None:
        create_summaries()

    global previous_step
    global previous_summary_time
    current_time = time.time()

    feed_dict = {k: np.mean(values[k]) for k in values}
    for var_name in images:
        feed_dict[var_name] = images[var_name]
    feed_dict["Steps/s"] = -1 if previous_step is None else (current_step - previous_step) / (current_time - previous_summary_time)

    summary = sess.run(
        merged,
        {placeholders[k]: feed_dict[k] for k in feed_dict}
    )
    train_writer.add_summary(summary, current_step)

    for var_name in values:
        values[var_name] = []

    previous_step = current_step
    previous_summary_time = current_time
