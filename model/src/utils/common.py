from official.nlp import optimization
from utils.config_loader import load_global_config

import tensorflow as tf
import tensorflow_text as text

# 从配置文件加载全局变量
global_config = load_global_config()
learning_rate = float(global_config.get('learning_rate', 0.001))
epochs = global_config.get('epochs', 32)

# 创建加速器
def create_adamw_optimizer(train_ds):
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=learning_rate,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
    return optimizer

def evaluate_model(test_ds, is_plot):
    saved_model_path = '/mnt/workspace/models/1'
    classifier_model = tf.saved_model.load(saved_model_path)
    loss, accuracy = classifier_model.evaluate(test_ds)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    if is_plot:
        draw()

def draw():
    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # r is for "solid red line"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

def test_text(model, text_list):
    results = tf.sigmoid(model(tf.constant(text_list)))
    

    print('Results from the saved model:')
    print_list(text_list, reloaded_results)
    print('Results from the model in memory:')
    print_list(text_list, original_results)

def print_list(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()