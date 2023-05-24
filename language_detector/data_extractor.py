from nltk.corpus import udhr
from nltk import sent_tokenize
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import re

class DataExtractor:
    def __init__(self, languages: dict):
        self.languages = languages

    def create_dict(self) -> dict:
        dictionary = {}
        value = 1
        for lang in self.languages: 
            raw = udhr.raw(lang["lang"])
            for letter in raw.lower(): 
                if letter not in dictionary.keys():
                    dictionary[letter] = value
                    value += 1
            lang["text"] = raw.lower()
        return dictionary
            
    def hot_encode(self): 
        labels = np.array([entry["lang"] for entry in self.languages]).reshape(-1,1)
        ohe = LabelBinarizer().fit(labels)
        y = ohe.transform(labels)
        for lang, label in zip(self.languages, y): 
            lang["ohe_label"] = list(label)

    
    def process_data(self, maxlen=25): 
        dictionary = self.create_dict()
        self.hot_encode()
        X, y = [], []
        for lang in self.languages: 
            text = lang["text"]
            for sent in sent_tokenize(text): 
                sent = [l for l in sent if l.isalpha()]
                for l, i in zip(sent, range(len(sent))):
                    if l not in dictionary: 
                        sent[i] = 0
                    else: 
                        sent[i] = dictionary[l]
                # padding
                if len(sent) > maxlen: 
                    sent = sent[:maxlen]
                else: 
                    difference = maxlen - len(sent)
                    for i in range(difference): 
                        sent.append(0)
                X.append(sent)
                y.append(lang["ohe_label"])
        return X, y
            
    def process_text(self, text: str, maxlen=25): 
        dictionary = self.create_dict()
        text = [l.lower() for l in text if l.isalpha()]
        for l, i in zip(text, range(len(text))):
            if l not in dictionary: 
                text[i] = 0
            else: 
                text[i] = dictionary[l]
        # padding
        if len(text) > maxlen: 
            text = text[:maxlen]
        else: 
            difference = maxlen - len(text)
            for i in range(difference): 
                text.append(0)
        return text
            
    def convert_prediction_to_langname(self, prediction: int) -> str: 
        self.hot_encode()
        split_symbols = r"[-_]"
        zeros_list = [0] * 28 # 29 languages
        zeros_list[prediction] = 1
        for entry in self.languages: 
            if entry["ohe_label"] == zeros_list: 
                lang = re.split(split_symbols, entry["lang"])[0]
                return lang
            
    def convert_ohe_to_index(self, ohe: list) -> int: 
        return ohe.index(1)