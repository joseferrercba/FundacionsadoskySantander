import nltk
import pandas as pd
from nltk.tokenize import regexp_tokenize, PunktSentenceTokenizer, RegexpTokenizer
from spellchecker import SpellChecker
import string
import re
from nltk.stem import SnowballStemmer
#from pattern.es import conjugate, INFINITIVE
import spacy

class CustomTokenizer(object):           
        
    def __init__(self):
        self.stemmer = SnowballStemmer('spanish')
        self.spell = SpellChecker(language='es')
        self.regexpTokenizer = RegexpTokenizer(r'\w+')
        self.punktSentenceTokenizer = PunktSentenceTokenizer()
        self.stopwords = nltk.corpus.stopwords.words('spanish')
        self.table = str.maketrans('', '', string.punctuation) 
        self.nlp = spacy.load("es_core_news_md")
        self.otherwords = ['eramos', 'estabamos', 'estais', 'estan', 'estara', 'estaran', 'estaras', 'estare', 'estareis', 'estaria', 'estariais', 'estariamos', 'estarian', 'estarias', 'esteis', 'esten', 'estes', 'estuvieramos', 'estuviesemos', 'fueramos', 'fuesemos', 'habeis', 'habia', 'habiais', 'habiamos', 'habian', 'habias', 'habra', 'habran', 'habras', 'habre', 'habreis', 'habria', 'habriais', 'habriamos', 'habrian', 'habrias', 'hayais', 'hubieramos', 'hubiesemos', 'mas', 'mia', 'mias', 'mio', 'mios', 'seais', 'sera', 'seran', 'seras', 'sere', 'sereis', 'seria', 'seriais', 'seriamos', 'serian', 'serias', 'si', 'tambien', 'tendra', 'tendran', 'tendras', 'tendre', 'tendreis', 'tendria', 'tendriais', 'tendriamos', 'tendrian', 'tendrias', 'teneis', 'tengais', 'tenia', 'teniais', 'teniamos', 'tenian', 'tenias', 'tuvieramos', 'tuviesemos']
        self.reservedWords = ['superclub', 'pagomiscuentas', 'americanexpress', 'cer', 'chubut', 'gmail']
      
    def __call__(self, sentence): 
        return self.processAll(sentence)
    
    def word_tokenize(self, sentence):
        words = nltk.word_tokenize(sentence)
        return words
    
    def removeAccents(self, word):        
        repl = {'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a',
                'é': 'e', 'ê': 'e',
                'í': 'i',
                'ó': 'o', 'ô': 'o', 'õ': 'o',
                'ú': 'u', 'ü': 'u',                
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
    
    # def conjugate_verb(self, word):
    #     doc = self.nlp(word)        
    #     #print([(w.text, w.pos_) for w in doc])            
    #     if doc[0].pos_ == 'VERB':
    #         word = conjugate(word, INFINITIVE)        
    #     return word
    
    def getFreqDist(self,sentence):
        freq_dist = nltk.FreqDist(sentence)     
        freq_df = pd.DataFrame(list(freq_dist.items()), columns = ["Word","Frequency"])
        print('FreqDist')
        print(freq_df.sort_values(by='Frequency', ascending=False))
        
    def removeNumbers(self, sentence):
        sentence = re.sub(r'\d+', '', sentence)
        return sentence
    
    def spell_correction(self, word):  
        if word not in self.reservedWords:
            word  = self.spell.correction(word)
        return word
    
    def spell_correction_reserved_word(self, word):
        if (word.startswith('aad') == True) | (word.startswith('advanta') == True) | (word.startswith('addvan') == True):
            word = 'aadvantage'    
        if (word.startswith('homeba') == True):
            word = 'homebanking'            
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
        return sentence
    
    def processAll(self,sentence):
        
        #spell_correction_reserved_word_in_sentence
        sentence = self.spell_correction_reserved_word_in_sentence(sentence)
        
        #Remove Numbers
        sentence = self.removeNumbers(sentence)
        
        #Remove Punctuation        
        words = self.removePunc(sentence)
        
        #Remove Stop Words                    
        words = self.removeStopWords(words)
        
        #spell_correction_reserved_word
        words = [self.spell_correction_reserved_word(w) for w in words]
        
        #spell_correction
        #words = [self.spell_correction(w) for w in words]
        
        #Remove Special Characters              
        words = [self.removeSpecialCharacters(w) for w in words]                      

        #conjugate_verb
        #words = [self.conjugate_verb(w) for w in words]     
        
        #Remove Accents                
        words = [self.removeAccents(w) for w in words]

        #Stem Words
        words = [self.stemmer.stem(w) for w in words]            

        return words
