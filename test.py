import tensorflow as tf
import tensorflow_datasets
from transformers import glue_convert_examples_to_features, TFBertForSequenceClassification, BertTokenizer
from transformers import (TFBertModel, )

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")

data = tensorflow_datasets.load("glue/mrpc")
train_dataset = data["train"]
train_dataset = glue_convert_examples_to_features(train_dataset, tokenizer, 128, 'mrpc')

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.fit(train_dataset, epochs=3, batch_size=16)
