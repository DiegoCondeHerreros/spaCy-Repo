import spacy
from spacy.training import Example
from spacy.scorer import Scorer
nlp = spacy.load("en_core_web_sm")
#Cargamos los datos de conll2003
raw_annotations=open("test.txt").read()
split_annotations=raw_annotations.split()
#Agrupamos los datos de entidades nombradas en tuplas
for n,i in enumerate(split_annotations):
    if i == "I-PER":    
        split_annotations[n]="I-PERSON"
    if i == "B-PER":
        split_annotations[n]="B-PERSON"    
    if i == "I-ORG":
        split_annotations[n]="I-ORGANIZATION"
    if i == "B-ORG":
        split_annotations[n]="B-ORGANIZATION"
    if i == "I-LOC":
        split_annotations[n]="I-LOCATION"
    if i == "B-LOC":
        split_annotations[n]="B-LOCATION"

#print(reference_annotations)
#Limpiamos los datos para usarlo con el clasificador NER
pure_tokens=split_annotations[::4]
tags=split_annotations[1::4]
pos_tags=split_annotations[2::4]
ner_tags=split_annotations[3::4]
#Convertimos los tokens en texto en bruto para pasarlo a la pipeline de spacy
raw_text=""
for i in range(0,len(pure_tokens)):
    raw_text += " "
    raw_text += pure_tokens[i]
#Obtenemos etiquetado de EN
doc=nlp(raw_text)
for ent in doc.ents:
    print(ent.text,ent.label_)
print(ner_tags)
#Creamos objetos tipo Example para obtener la evaluación de las EN
example = Example.from_dict(doc,{"text":raw_text,"words":pure_tokens,"pos":tags,"tags":pos_tags,"entities":ner_tags})
scorer=Scorer()
scores=scorer.score([example])
print(scores)
print("-------------------------------------")
print("Precision NER")
print(scores["ents_p"])
print("Recall NER")
print(scores["ents_r"])
print("F-1 NER")
print(scores["ents_f"])
print("--------------------------------------")
print("Token Accuracy")
print(scores["token_acc"])
print("Token Precision")
print(scores["token_p"])
print("Token Recall")
print(scores["token_r"])
print("Token F")
print(scores["token_f"])
print("---------------------------------------")
print("Sentence Precision")
print(scores["sents_p"])
print("Sentence Recall")
print(scores["sents_r"])
print("Sentence F")
print(scores["sents_f"])
print("-------------------------------------------")
print("Accuracy POS finegrained")
print(scores["tag_acc"])
print("Accuracy POS biggrained")
print(scores["pos_acc"])
print("Accuracy Morphological")
print(scores["morph_acc"])

