from re import error
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from enum import Enum, unique
@unique
class VectEnum(Enum):
    TfidfVectorizer = 1
    CountVectorizer = 2

class Vectorizer(object):
    """
    Return vector to transform dataset
    """
    
    def get_vectorizer(self, vectorizer_type = VectEnum.TfidfVectorizer, min_df=1, tokenizer_type=None):          
        if vectorizer_type.name == VectEnum.TfidfVectorizer.name:
            return TfidfVectorizer(min_df=min_df, 
                                   tokenizer=tokenizer_type)
        if vectorizer_type.name == VectEnum.CountVectorizer.name:
            return CountVectorizer(min_df=min_df, 
                                   tokenizer=tokenizer_type)   
        raise error('Theres is no vect configured')
