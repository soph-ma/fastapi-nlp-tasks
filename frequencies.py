from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def create_freq_dist(text): 
    fdist = FreqDist(word.lower() for word in word_tokenize(text)).most_common()
    return fdist