#!/usr/bin/env python3
# from: https://github.com/yk/huggingface-nlp-demo/blob/master/demo.py

from absl import app, flags, logging

import torch as th
import pytorch_lightning as pl

import nlp
import transformers

import os
from pathlib import Path
from datetime import datetime

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 1, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('percent', 5, '')

class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)

        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(
                    x['text'], 
                    max_length=FLAGS.seq_length, 
                    pad_to_max_length=True)['input_ids']
            return x

        def _prepare_ds(split):
            ds = nlp.load_dataset('imdb', split=f'{split}')  #[:{FLAGS.batch_size if FLAGS.debug else f"{FLAGS.percent}%"}]
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.test_ds, self.validation_ds = map(_prepare_ds, ('train', 'test', 'unsupervised'))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=FLAGS.batch_size,
                drop_last=True,
                shuffle=True,
                )

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        print(f'val loss: {loss} - val acc:{acc}')                
        return {**out, 'log': out}

    def val_dataloader(self):
        return th.utils.data.DataLoader(self.test_ds,
                                        batch_size=FLAGS.batch_size,
                                        drop_last=False,
                                        shuffle=False,
                                        )


    def test_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'test_loss': loss, 'test_acc': acc}


    def test_epoch_end(self, outputs):
        loss = th.cat([o['test_loss'] for o in outputs], 0).mean()
        acc = th.cat([o['test_acc'] for o in outputs], 0).mean()
        out = {'test_loss': loss, 'test_acc': acc}
        print(f'test loss: {loss} - test acc:{acc}')        
        return {**out, 'log': out}

    def test_dataloader(self):
        return th.utils.data.DataLoader(
                self.validation_ds,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=False,
                )

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
        )


def main(_):
    model = IMDBSentimentClassifier()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=datetime.now().strftime("%Y%m%d-%H%M%S")),
    )
    trainer.fit(model)
    trainer.test()


FLAGS = flags.FLAGS

if not Path('logs').exists():
    os.mkdir('logs')


if __name__ == '__main__':
    app.run(main)