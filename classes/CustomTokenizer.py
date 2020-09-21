from __future__ import generator_stop
import nltk
import pandas as pd
from nltk.tokenize import regexp_tokenize, PunktSentenceTokenizer, RegexpTokenizer
#from spellchecker import SpellChecker
import string
import re
from nltk.stem import SnowballStemmer
from pattern.text.es import conjugate, INFINITIVE
import spacy
from textacy import preprocessing

stemmer = SnowballStemmer('spanish')
regexpTokenizer = RegexpTokenizer(r'\w+')
punktSentenceTokenizer = PunktSentenceTokenizer()
stopwords = nltk.corpus.stopwords.words('spanish')
table = str.maketrans('', '', string.punctuation) 
nlp = spacy.load("es_core_news_md")
otherwords = ['eramos', 'estabamos', 'estais', 'estan', 'estara', 'estaran', 'estaras', 'estare', 'estareis', 'estaria', 'estariais', 'estariamos', 'estarian', 'estarias', 'esteis', 'esten', 'estes', 'estuvieramos', 'estuviesemos', 'fueramos', 'fuesemos', 'habeis', 'habia', 'habiais', 'habiamos', 'habian', 'habias', 'habra', 'habran', 'habras', 'habre', 'habreis', 'habria', 'habriais', 'habriamos', 'habrian', 'habrias', 'hayais', 'hubieramos', 'hubiesemos', 'mas', 'mia', 'mias', 'mio', 'mios', 'seais', 'sera', 'seran', 'seras', 'sere', 'sereis', 'seria', 'seriais', 'seriamos', 'serian', 'serias', 'si', 'tambien', 'tendra', 'tendran', 'tendras', 'tendre', 'tendreis', 'tendria', 'tendriais', 'tendriamos', 'tendrian', 'tendrias', 'teneis', 'tengais', 'tenia', 'teniais', 'teniamos', 'tenian', 'tenias', 'tuvieramos', 'tuviesemos']
reservedWords = ['superclub', 'pagomiscuentas', 'americanexpress', 'cer', 'chubut', 'gmail', 'americanairlines', 'cbu']
        
def remove_accents(word):        
    repl = {'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
            'é': 'e', 'ê': 'e',
            'í': 'i',
            'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
            'ú': 'u', 'ü': 'u',
            'ç': 'c',
            'π': 'pi', 'º': '', 'æ': '',
            'Ã©': 'e','Ã³': 'o','Ã±': 'i','Ãº': 'u','Ã': 'a'
           }
    new_word = ''.join([repl[c].lower() if c in repl else c for c in word])
    return new_word

def listToString(sentence):  
    # initialize an empty string 
    str1 = ""  
    # traverse in the string   
    for ele in sentence:  
        str1 = str1.strip() + ' ' + ele.strip().lower()   
    # return string   
    return str1  

def remove_stopwords(words):                
    words = [word for word in words if word not in stopwords]
    words = [word for word in words if word not in otherwords]
    return words

def remove_special_characters(word):                     
    word = word.translate(table)
    return word

def remove_punc(sentence):        
    words = punktSentenceTokenizer.tokenize(sentence)[0]
    words = regexpTokenizer.tokenize(words)
    return words

def removeWhitespaces(sentence):
    sentence = str(sentence).strip() if not pd.isna(sentence) else ''
    return sentence

def conjugate_verb(word):     
    #print([(w.text, w.pos_) for w in doc])            
    if nlp(word)[0].pos_ == 'VERB':
        word = conjugate(word, INFINITIVE)        
    return word

def conjugate_verb_in_sentence(sentence):     
    words = [conjugate_verb(w) for w in sentence]    
    return words

def getFreqDist(sentence):
    freq_dist = nltk.FreqDist(sentence)     
    freq_df = pd.DataFrame(list(freq_dist.items()), columns = ["Word","Frequency"])
    print('FreqDist')
    print(freq_df.sort_values(by='Frequency', ascending=False))
    
def remove_numbers(sentence):
    sentence = re.sub(r'\d+', '', sentence)
    return sentence

def spell_correction_reserved_word(word):
    if (word.startswith('aadnatage') == True) | (word.startswith('aad') == True) | (word.startswith('advanta') == True) | (word.startswith('addvan') == True):
        word = 'advantage'    
    if (word.startswith('homeba') == True):
        word = 'homebanking'            
    if (word.startswith('aaños') == True):
        word = 'aaños'
    if (word == 'aa') | (word == 'aan'):
        word = 'advantage'
    if (word == 'aap'):
        word = 'app'
    if (word == 'x'):
        word = 'por'        
    if (word == 'q'):
        word = 'que'    
    if (word == 'ago'):
        word = 'hago'        
    return word

def spell_correction_reserved_word_in_sentence(sentence):
    if ('super club' in sentence) == True:
        sentence = sentence.replace('super club', 'superclub')
    if ('pago mis cuentas' in sentence) == True:
        sentence = sentence.replace('pago mis cuentas', 'pagomiscuentas')
    if ('american express' in sentence) == True:        
        sentence = sentence.replace('american express', 'americanexpress')
    if ('toquen' in sentence) == True:
        sentence = sentence.replace('toquen', 'token')
    if ('american airlines' in sentence) == True:
        sentence = sentence.replace('american airlines', 'americanairlines')
    if ('infinity platimun' in sentence) == True:
        sentence = sentence.replace('infinity platimun', 'infinityplatimun')
    if ('platimun' in sentence) == True:
        sentence = sentence.replace('platimun', 'infinityplatimun')
    if ('black' in sentence) == True:
        sentence = sentence.replace('black', 'tarjetablack')
    if ('tarjeta black' in sentence) == True:
        sentence = sentence.replace('tarjeta black', 'tarjetablack')
    if ('plazo fijo' in sentence) == True:
        sentence = sentence.replace('plazo fijo', 'plazofijo')
    if ('clave' in sentence) == True:
        sentence = sentence.replace('clave', 'clavebanco')
    if ('adherir servicios' in sentence) == True:
        sentence = sentence.replace('adherir servicios', 'adherirservicios')
    if ('debitar' in sentence) == True:
        sentence = sentence.replace('debitar', 'debitoautomatico')
    if ('debito' in sentence) == True:
        sentence = sentence.replace('debito', 'debitoautomatico')
    if ('debito automatico' in sentence) == True:
        sentence = sentence.replace('debito automatico', 'debitoautomatico')
    if ('debitar automatico' in sentence) == True:
        sentence = sentence.replace('debitar automatico', 'debitoautomatico')
    if ('home banking' in sentence) == True:
        sentence = sentence.replace('home banking', 'homebanking')
        
    return sentence

def textacy_preprocess(sentence):
    """Preprocess text."""
    sentence = preprocessing.normalize_hyphenated_words(sentence)
    sentence = preprocessing.normalize_quotation_marks(sentence)
    #sentence = preprocessing.normalize_repeating_chars(sentence)
    sentence = preprocessing.normalize_unicode(sentence)
    sentence = preprocessing.normalize_whitespace(sentence)
    sentence = preprocessing.remove_accents(sentence)
    sentence = preprocessing.remove_punctuation(sentence)
    sentence = preprocessing.replace_currency_symbols(sentence)
    sentence = preprocessing.replace_emails(sentence)
    sentence = preprocessing.replace_emojis(sentence)
    sentence = preprocessing.replace_hashtags(sentence)
    sentence = preprocessing.replace_numbers(sentence)
    sentence = preprocessing.replace_phone_numbers(sentence)
    sentence = preprocessing.replace_urls(sentence)
    sentence = preprocessing.replace_user_handles(sentence)
    
    return sentence

def custom_preprocess(sentence, removeNumbers = True, removePunc = True, 
                removeStopWords = True, spell_correction_reservedword=True,
                spell_correction_reservedword_in_sentence = True,
                removeSpecialCharacters = True, removeAccents = True, stem = True, 
                conjugate_verbs = False):
    words = []
    if spell_correction_reservedword_in_sentence == True:
        #spell_correction_reserved_word_in_sentence
        sentence = spell_correction_reserved_word_in_sentence(sentence)
    
    if removeNumbers == True:
        #Remove Numbers
        sentence = remove_numbers(sentence)
    
    if removePunc == True:
        #Remove Punctuation        
        words = remove_punc(sentence)
    
    if removeStopWords == True:
        #Remove Stop Words                    
        words = remove_stopwords(words)
    
    if spell_correction_reservedword == True:
        #spell_correction_reserved_word
        words = [spell_correction_reserved_word(w) for w in words]        
    
    if removeSpecialCharacters == True:
        #Remove Special Characters              
        words = [remove_special_characters(w) for w in words]      
                        
    if conjugate_verbs == True:
        #conjugate_verb
        words = [conjugate_verb(w) for w in words]     
    
    if removeAccents == True:
        #Remove Accents                
        words = [remove_accents(w) for w in words]
    if stem == True:
        #Stem Words
        words = [stemmer.stem(w) for w in words]            
    return words
