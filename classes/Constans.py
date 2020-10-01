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
TOKENIZER_TYPE = None #custom_preprocess | nltk.word_tokenize
SHUFFLE = True
SAMPLING_STRATEGY = 'minority'
N_JOBS = -1
K_NEIGHBORS = 4
SCORING = 'balanced_accuracy'
REFIT = 'balanced_accuracy'
CV = 5
COLUMNA_PREGUNTAS = 'Preguntas_word_tokenize' 
VERBOSE = 0
CLASS_WEIGHT = 'balanced'
# Choose between following columns: 
#-----------------------------------------
# Preguntas_custom_preprocess_no_stopwords
# Preguntas_custom_preprocess
# Preguntas_custom_preprocess_w_verbs
# Preguntas_word_tokenize
