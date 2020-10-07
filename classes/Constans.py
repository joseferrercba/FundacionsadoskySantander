import nltk
from classes.Vectorizer import VectEnum
from classes.Resample import ResamplerEnum
#--------------------------------------------------#
###                  PARAMETERS                  ###
#--------------------------------------------------#
APPLY_RESAMPLE = True
VECTORIZER_TYPE = VectEnum.TfidfVectorizer
RESAMPLER_TYPE = ResamplerEnum.SMOTE
RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_DF = 1            
TOKENIZER_TYPE = nltk.word_tokenize #custom_preprocess | nltk.word_tokenize
SHUFFLE = True
SAMPLING_STRATEGY = 'minority'
N_JOBS = -1
K_NEIGHBORS = 4
SCORING = 'balanced_accuracy'
REFIT = 'balanced_accuracy'
CV = 5
COLUMNA_PREGUNTAS = 'Pregunta' 
VERBOSE = 0
CLASS_WEIGHT = 'balanced'
EPOCHS = 50
BATCH_SIZE = 64
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 100
# This is fixed.
EMBEDDING_DIM = 100
# Choose between following columns: 
#-----------------------------------------
# Preguntas_custom_preprocess_no_stopwords
# Preguntas_custom_preprocess
# Preguntas_custom_preprocess_w_verbs
# Preguntas_word_tokenize
# Preguntas_en
# Preguntas_es
