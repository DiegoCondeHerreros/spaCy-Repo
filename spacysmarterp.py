import spacy
from spacy.training import Example
from spacy.scorer import Scorer
#ESTA SECCION DEL CODIGO SE EJECUTA DESPUES DE LA CREACION DE NLP
@spacy.registry.callbacks("procesado_nlp")
def create_callback():
    def procesado_nlp(nlp):       
        known_entities=["B-MISC","I-MISC","O"]
        for n,i in enumerate(nlp.ent.label_):
            if i not in known_entities:
                if i.startswith("B-"):
                    nlp.ent.label_[n]="B-MISC"
                if i.startswith("I-"):
                    nlp.ent.label_[n]="I-MISC"
