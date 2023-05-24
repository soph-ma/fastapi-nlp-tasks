import heapq
import nltk
from nltk import sent_tokenize, word_tokenize

class Summarizer: 
    def __init__(self, text: str):
        self.text = text 
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.sents = sent_tokenize(text)
        self.words = word_tokenize(text.lower())

    # def preprocess(self):
    #     with open("linking_words.txt", "r") as f: 
    #         linkinig_words = f.readlines()
    #     for sent in self.sents:
    #         for w in linkinig_words: 
    #             if sent.lower().startswith(w):
    #                 word_length = len(w) 
    #                 sent.replace(sent[:len], "")
                    

    def get_weighted_frequencies(self) -> dict: 
        frequencies = {}
        weighted_frequencies = {}
        for word in self.words:
            if word not in self.stopwords:
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
        highest = max(frequencies.values())
        for word in frequencies.keys():
            weighted_frequencies[word] = (frequencies[word]/highest)
        
        return weighted_frequencies

    def calculate_sent_scores(self) -> dict:
        sent_scores = {}
        weighted_frequencies = self.get_weighted_frequencies()
        for sent in self.sents:
            for word in word_tokenize(sent.lower()):
                if word in weighted_frequencies.keys():
                    if sent not in sent_scores.keys():
                        sent_scores[sent] = weighted_frequencies[word]
                    else:
                        sent_scores[sent] += weighted_frequencies[word] 

        return sent_scores

    def summarize(self, num_sents: int = 7) -> str:
        sent_scores = self.calculate_sent_scores()
        top_sentences = heapq.nlargest(num_sents, sent_scores, key=sent_scores.get)
        summary = ' '.join(top_sentences)

        return summary
            


