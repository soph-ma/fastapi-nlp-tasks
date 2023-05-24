from config import LANGUAGES
from data_extractor import DataExtractor
from training import prediction

extractor = DataExtractor(LANGUAGES)
data = extractor.process_data()
