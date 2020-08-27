from __future__ import generator_stop
import nltk
import pandas as pd
from nltk.tokenize import regexp_tokenize, PunktSentenceTokenizer, RegexpTokenizer
#from spellchecker import SpellChecker
import string
import re
from nltk.stem import SnowballStemmer
from pattern.text.es import conjugate, INFINITIVE
from autocorrect import Speller
import spacy
from textacy import preprocessing

class CustomTokenizer(object):           
        
    def __init__(self):
        self.stemmer = SnowballStemmer('spanish')
        self.spell = Speller('es')
        self.regexpTokenizer = RegexpTokenizer(r'\w+')
        self.punktSentenceTokenizer = PunktSentenceTokenizer()
        self.stopwords = nltk.corpus.stopwords.words('spanish')
        self.table = str.maketrans('', '', string.punctuation) 
        self.nlp = spacy.load("es_core_news_md")
        self.otherwords = ['eramos', 'estabamos', 'estais', 'estan', 'estara', 'estaran', 'estaras', 'estare', 'estareis', 'estaria', 'estariais', 'estariamos', 'estarian', 'estarias', 'esteis', 'esten', 'estes', 'estuvieramos', 'estuviesemos', 'fueramos', 'fuesemos', 'habeis', 'habia', 'habiais', 'habiamos', 'habian', 'habias', 'habra', 'habran', 'habras', 'habre', 'habreis', 'habria', 'habriais', 'habriamos', 'habrian', 'habrias', 'hayais', 'hubieramos', 'hubiesemos', 'mas', 'mia', 'mias', 'mio', 'mios', 'seais', 'sera', 'seran', 'seras', 'sere', 'sereis', 'seria', 'seriais', 'seriamos', 'serian', 'serias', 'si', 'tambien', 'tendra', 'tendran', 'tendras', 'tendre', 'tendreis', 'tendria', 'tendriais', 'tendriamos', 'tendrian', 'tendrias', 'teneis', 'tengais', 'tenia', 'teniais', 'teniamos', 'tenian', 'tenias', 'tuvieramos', 'tuviesemos']
        self.reservedWords = ['superclub', 'pagomiscuentas', 'americanexpress', 'cer', 'chubut', 'gmail', 'americanairlines']
      
    def __call__(self, sentence): 
        return self.processAll(sentence)
    
    def word_tokenize(self, sentence):
        words = nltk.word_tokenize(sentence)
        return words
    
    def removeAccents(self, word):        
        repl = {'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
                'é': 'e', 'ê': 'e',
                'í': 'i',
                'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
                'ú': 'u', 'ü': 'u',
                'ç': 'c',
                'π': 'pi', 'º': '', 'æ': ''                
               }

        new_word = ''.join([repl[c].lower() if c in repl else c for c in word])
        return new_word
    
    def listToString(self,sentence):  
        # initialize an empty string 
        str1 = ""  
        # traverse in the string   
        for ele in sentence:  
            str1 = str1.strip() + ' ' + ele.strip().lower()   
        # return string   
        return str1  
    
    def removeStopWords(self, words):                
        words = [word for word in words if word not in self.stopwords]
        words = [word for word in words if word not in self.otherwords]
        return words
    
    def removeSpecialCharacters(self, word):                     
        word = word.translate(self.table)
        return word
    
    def removePunc(self,sentence):        
        words = self.punktSentenceTokenizer.tokenize(sentence)[0]
        words = self.regexpTokenizer.tokenize(words)
        return words
    
    def removeWhitespaces(self,sentence):
        sentence = str(sentence).strip() if not pd.isna(sentence) else ''
        return sentence
    
    def conjugate_verb(self, word):     
        #print([(w.text, w.pos_) for w in doc])            
        if self.nlp(word)[0].pos_ == 'VERB':
            word = conjugate(word, INFINITIVE)        
        return word
    
    def getFreqDist(self,sentence):
        freq_dist = nltk.FreqDist(sentence)     
        freq_df = pd.DataFrame(list(freq_dist.items()), columns = ["Word","Frequency"])
        print('FreqDist')
        print(freq_df.sort_values(by='Frequency', ascending=False))
        
    def removeNumbers(self, sentence):
        sentence = re.sub(r'\d+', '', sentence)
        return sentence
    
    def spell_correction(self, sentence):  
        #if sentence not in self.reservedWords:
        sentence = self.spell(sentence)
        return sentence
    
    def spell_correction_reserved_word(self, word):
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
    
    def spell_correction_reserved_word_in_sentence(self, sentence):
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
        return sentence
    
    def textacy_preprocess(self, sentence, fix_unicode=False, lowercase=True,
                    no_urls=False, no_emails=False,
                    no_phone_numbers=False,
                    no_numbers=False, no_currency_symbols=False,
                    no_punct=True, no_accents=True):
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

    def custom_preprocess(self, sentence, removeNumbers = True, removePunc = True, 
                    removeStopWords = True, spell_correction_reserved_word=True,
                    spell_correction_reserved_word_in_sentence = True,
                    removeSpecialCharacters = True, removeAccents = True, stem = True, 
                    conjugate_verb = True, spell_correction = True):
        words = []
        if spell_correction_reserved_word_in_sentence == True:
            #spell_correction_reserved_word_in_sentence
            sentence = self.spell_correction_reserved_word_in_sentence(sentence)
        
        if spell_correction == True:
            #spell_correction
            sentence = self.spell_correction(sentence)

        if removeNumbers == True:
            #Remove Numbers
            sentence = self.removeNumbers(sentence)
        
        if removePunc == True:
            #Remove Punctuation        
            words = self.removePunc(sentence)
        
        if removeStopWords == True:
            #Remove Stop Words                    
            words = self.removeStopWords(words)
        
        if spell_correction_reserved_word == True:
            #spell_correction_reserved_word
            words = [self.spell_correction_reserved_word(w) for w in words]        
        
        if removeSpecialCharacters == True:
            #Remove Special Characters              
            words = [self.removeSpecialCharacters(w) for w in words]                      

        if conjugate_verb == True:
            #conjugate_verb
            words = [self.conjugate_verb(w) for w in words]     
        
        if removeAccents == True:
            #Remove Accents                
            words = [self.removeAccents(w) for w in words]

        if stem == True:
            #Stem Words
            words = [self.stemmer.stem(w) for w in words]            

        return words
