#taken from https://blog.tensorflow.org/2019/11/hugging-face-state-of-art-natural.html

import tensorflow as tf
import tensorflow_datasets
from transformers import glue_convert_examples_to_features
from transformers import TFBertModel, BertTokenizer

model = TFBertModel.from_pretrained("bert-base-cased")  # Automatically loads the config
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

print(model.config)

data = tensorflow_datasets.load("glue/mrpc")

train_dataset = data["train"]
validation_dataset = data["validation"]

train_dataset = glue_convert_examples_to_features(train_dataset, tokenizer, 128, 'mrpc')
validation_dataset = glue_convert_examples_to_features(validation_dataset, tokenizer, 128, 'mrpc')
#train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
#validation_dataset = validation_dataset.batch(64)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = model.fit(train_dataset, 
                    epochs=2, 
                    steps_per_epoch=115, 
                    validation_data=validation_dataset, 
                    validation_steps=7
                    )
