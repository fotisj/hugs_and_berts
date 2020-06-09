import ktrain
from ktrain import text
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'

train_data, test_data, preproc = text.texts_from_folder('aclImdb', classes=['neg', 'pos'],
                                                                   maxlen=500, 
                                                                   preprocess_mode='distilbert', 
                                                                   train_test_names=['train', 'test'], 
                                                                   encoding='utf-8',
                                                                   lang='en'
                                                                   )

model = text.text_classifier('distilbert', train_data, preproc=preproc)

learner = ktrain.get_learner(model, 
                            train_data=train_data, 
                            val_data=test_data, 
                            batch_size=6)

learner.fit_onecycle(2e-5, 1)

predictor = ktrain.get_predictor(model, preproc)
print('Saving model...')
predictor.save('imdb_classifier.bin')

text = "Some of the characters were really charming, but the plot was slightly absurd. And I atleast have never seen \
        a butler doing something like this. All in all quite a waste of time."
r = predictor.predict(text)

print(f'prediction for text (>{text[:30]}< : {r} sentiment')

data = [ 'This movie was horrible! The plot was boring. Acting was okay, though.',
         'The film really sucked. I want my money back.',
        'What a beautiful romantic comedy. 10/10 would see again!']

print(data)
print(predictor.get_classes())
print(predictor.predict(data, return_proba=True))





