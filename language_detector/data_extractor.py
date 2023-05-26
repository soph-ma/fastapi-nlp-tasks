from nltk.corpus import udhr, swadesh
from nltk import sent_tokenize
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random
import re

class DataExtractor:
    def __init__(self, languages: dict):
        self.languages = languages
        self.dictionary = self._create_dict()

    def _enlarge_text(self, text: str, num_sentences: int = 120000) -> str:
        """ Artificially creates more sentences for a given text """
        words = text.split()
        word_dict = {}

        # building a dictionary of word transitions
        for i in range(len(words) - 1):
            if words[i] not in word_dict:
                word_dict[words[i]] = []
            word_dict[words[i]].append(words[i+1])

        new_text = []
        current_word = random.choice(words)
        new_text.append(current_word)

        # generating new sentences based on the Markov chain transitions
        while len(new_text) < num_sentences:
            if current_word not in word_dict:
                current_word = random.choice(words)
                new_text.append(current_word)
            else:
                next_word = random.choice(word_dict[current_word])
                current_word = next_word
                new_text.append(next_word)
        return text + ' '.join(new_text)

    def _create_dict(self) -> dict:
        dictionary = {}
        value = 1
        for lang in self.languages: 
            if lang["swadesh"] == None:
                raw = udhr.raw(lang["lang"]).lower()
            else: 
                swadesh_words = swadesh.words(lang["swadesh"])
                swadesh_text = " ".join(([" ".join(swadesh_words[i:i+5]) + "." for i in range(0, len(swadesh_words), 5)]))
                raw = udhr.raw(lang["lang"]).lower() + swadesh_text
            for letter in raw: 
                if letter not in dictionary.keys():
                    dictionary[letter] = value
                    value += 1
            lang["text"] = raw
        return dictionary
            
    def hot_encode(self): 
        labels = np.array([entry["lang"] for entry in self.languages]).reshape(-1,1)
        ohe = LabelBinarizer().fit(labels)
        y = ohe.transform(labels)
        for lang, label in zip(self.languages, y): 
            lang["ohe_label"] = list(label)

    
    def process_data(self, maxlen=125): 
        self.hot_encode()
        X, y = [], []
        for lang in self.languages: 
            text = lang["text"]
            text = self._enlarge_text(text)
            if lang["lang"] == "Chinese_Mandarin-GB2312" or lang["lang"] == "Japanese_Nihongo-UTF8": # langs that do not use full stop for sentence division
                text = [text[i:i+125] for i in range(0, len(text), 125)] 
            else: text = sent_tokenize(text)
            for sent in text: 
                sent = [l for l in sent]
                for l, i in zip(sent, range(len(sent))):
                    if l not in self.dictionary: 
                        sent[i] = 0
                    else: 
                        sent[i] = self.dictionary[l]
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
            
    def process_text(self, text: str, maxlen=125): 
        text = [l.lower() for l in text]
        for l, i in zip(text, range(len(text))):
            if l not in self.dictionary: 
                text[i] = 0
            else: 
                text[i] = self.dictionary[l]
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
        zeros_list = [0] * 27 # 29 languages
        zeros_list[prediction] = 1
        for entry in self.languages: 
            if entry["ohe_label"] == zeros_list: 
                lang = re.split(split_symbols, entry["lang"])[0]
                return lang
            
    def convert_ohe_to_index(self, ohe: list) -> int: 
        return ohe.index(1)