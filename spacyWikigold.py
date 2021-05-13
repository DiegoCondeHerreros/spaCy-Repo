import spacy
from spacy.training import Example
from spacy.scorer import Scorer
nlp = spacy.load("en_core_web_sm")

#doc = nlp("Apple is looking at buying U.K startup for $1 billion")
#Cargamos los datos de Wikigold
raw_annotations = open("wikigold.conll.txt").read()
split_annotations=raw_annotations.split()
#Agrupamos los datos de entidades nombradas en tuplas
def group(lst,n):
    for i in range(0,len(lst),n):
        val=lst[i:i+n]
        if len(val) == n:
            yield tuple(val)
reference_annotations = list(group(split_annotations,2))
ref_dict=dict(reference_annotations)
print(reference_annotations)
#Limpiamos los datos para usarlo con el clasificador NER
pure_tokens=split_annotations[::2]
tags=split_annotations[1::2]
#Convertimos los tokens en texto en bruto para pasarlo a la pipeline de spacy
raw_text = ""
for i in range(0,len(pure_tokens)):
    raw_text += " "
    raw_text += pure_tokens[i]
#Obtenemos etiquetado de EN
doc=nlp(raw_text)
#Creamos objetos tipo Example para obtener la evaluacion de las EN
#example = Example(doc,reference)
example = Example.from_dict(doc, {"text":raw_text, "words":pure_tokens, "tags":tags, "entities":tags})
scorer = Scorer()
scores = scorer.score([example])
print("Precision")
print(scores["ents_p"])
print("Recall")
print(scores["ents_r"])
print("F-1")
print(scores["ents_f"])
print("Token Accuracy")
print(scores["token_acc"])
print("Token Precision")
print(scores["token_p"])
print("Token Recall")
print(scores["token_r"])
print("Token F")
print(scores["token_f"])
