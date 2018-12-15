import numpy as np
import tensorflow as tf
import time

path = None
previous_summary_time = None
previous_step = None

train_writer = None

images = {}
new_images = {}
values = {}
new_values = []
placeholders = {}


def init(summary_path):
    global path
    path = summary_path
    new_values.append("Stats/StepsPerSec")


def add(variable_name, value):
    global need_refresh
    if variable_name not in values:
        new_values.append(variable_name)
        values[variable_name] = [value]
        need_refresh = True
    else:
        values[variable_name].append(value)


def add_image(variable_name, value):
    new_images[variable_name] = value
    images[variable_name] = value


def create_summaries():
    global new_images
    global new_values
    for var_name in new_values:
        placeholders[var_name] = tf.placeholder(tf.float32, ())
        tf.summary.scalar(var_name, placeholders[var_name])
    new_values = []

    for var_name in new_images:
        placeholders[var_name] = tf.placeholder(tf.float32, np.shape(new_images[var_name]))
        tf.summary.image(var_name, placeholders[var_name])
    new_images = {}

    global merged
    merged = tf.summary.merge_all()
    global sess
    sess = tf.get_default_session()
    global train_writer
    if train_writer is None:
        train_writer = tf.summary.FileWriter(path)


def save(current_step):
    if need_refresh:
        create_summaries()

    if train_writer:
        global previous_step
        global previous_summary_time
        current_time = time.time()

        values_with_content = [values[k] for k in values if values[k]]
        # Don't save if some of the statistics don't have any content -> Avoid NaNs in totalReward summary
        if len(values_with_content) == len(values):
            feed_dict = {k: np.mean(values[k]) for k in values}
            for var_name in images:
                feed_dict[var_name] = images[var_name]
            feed_dict["Stats/StepsPerSec"] = -1 if previous_step is None else (current_step - previous_step) / (current_time - previous_summary_time)

            summary = sess.run(
                merged,
                {placeholders[k]: feed_dict[k] for k in feed_dict}
            )
            train_writer.add_summary(summary, current_step)
            train_writer.flush()

            for var_name in values:
                values[var_name] = []

            previous_step = current_step
            previous_summary_time = current_time
