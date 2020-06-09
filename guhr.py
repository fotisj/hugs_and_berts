"""
Testing the German sentiment Bert model by Oliver Guhr et al. 2020*
with excerpts of German poetry

model: https://huggingface.co/oliverguhr/german-sentiment-bert
repo: https://github.com/oliverguhr/german-sentiment
publication: http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.202.pdf
installation: pip install germansentiment
"""

import re
from germansentiment import SentimentModel



corpus = """Der erste Tag im Monat Mai
Ist mir der glücklichste von allen.
Dich sah ich und gestand dir frei,
Den ersten Tag im Monat Mai,
Daß dir mein Herz ergeben sei.
Wenn mein Geständnis dir gefallen,
So ist der erste Tag im Mai
Für mich der glücklichste von allen.
###
Denk ich an Deutschland in der Nacht,
Dann bin ich um den Schlaf gebracht,
Ich kann nicht mehr die Augen schließen,
Und meine heißen Tränen fließen.
###
Über sturmzerfetzter Wolken Widerschein,
In des toten Dunkels kalten Wüstenein,
Daß er mit dem Brande weit die Nacht verdorr,
Pech und Feuer träufet unten auf Gomorrh.
###
Ewig trägt im Mutterschoße,
Süße Königin der Flur!
Dich und mich die stille, große,
Allbelebende Natur;
Röschen! unser Schmuck veraltet,
Stürm entblättern dich und mich,
Doch der ewge Keim entfaltet
Bald zu neuer Blüte sich.
"""

labels = ['pos', 'neg', 'neg', 'pos']
corpus = re.sub('\n', ' ', corpus)
texts = corpus.split('###')

model = SentimentModel()
result = model.predict_sentiment(texts)

for text, sent, label in zip(texts, result, labels):
    print(f'"{text[:60]}" -> {sent} (richtig: {label})')