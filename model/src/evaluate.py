from utils.common import *
from utils.config_loader import load_global_config

global_config = load_global_config()
batch_size = global_config.get('batch_size', 32)

test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)
evaluate_model(test_ds, True)