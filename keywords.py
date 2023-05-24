import nltk
from nltk.util import ngrams
from keybert import KeyBERT

class KeyWordsExtractor:
    def __init__(self, text): 
        self.text = text
        self.stopwords = nltk.corpus.stopwords.words("english")
    
    def extract_ngrams(self) -> str: 
        ngrams_str = ""
        unigrams = list(self.text.split())
        bigrams = list(ngrams(unigrams, 2))
        trigrams = list(ngrams(unigrams, 3))

        ngrams_types = (unigrams, bigrams, trigrams)
        for ngram_type in ngrams_types: 
            for ngram in ngram_type: 
                ngram = " ".join(ngram[0].split())
                ngrams_str += (ngram +", ")

        return ngrams_str

    
    def extract_keywords(self):
        ngrams = self.extract_ngrams()
        kw_model = KeyBERT()
        kwords = kw_model.extract_keywords(ngrams, 
                                           keyphrase_ngram_range=(1, 1), 
                                           stop_words = "english",
                                           top_n = 10)
        return [kw[0] for kw in kwords]

