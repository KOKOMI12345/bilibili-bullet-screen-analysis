import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from utils.config_loader import load_global_config
from utils.common import create_adamw_optimizer

# 从配置文件加载全局变量
global_config = load_global_config()
epochs = global_config.get('epochs', 10)

def build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

def train_classifier_model(model, train_ds, val_ds):
    # 损失函数和评估指标
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    # 创建加速器
    optimizer = create_adamw_optimizer(train_ds)
    # 编译模型
    model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
    print(f'Training model...')
    history = model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)
    tf.saved_model.save(model, "/mnt/workspace/bilibili-bullet-screen-analysis/model/models/1")
    print("导出模型成功")
    

