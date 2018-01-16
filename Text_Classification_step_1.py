import numpy as np
import operator
import nltk
import string
import io
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
#from nltk.stem import SnowballStemmer
#from nltk.stem.porter import PorterStemmer
"""
This code is the first step to preprocess the text data.

download these files if you dont have them already.
#nltk.download('wordnet')
#nltk.download('stopwords')
"""

def filter_line(line,L,wnl,my_string_punct, stopwords):

    line = line.lower()
    for elemento in string.punctuation:
        line = line.replace(elemento," %s" % elemento)

    all_tokens = []
    all_tokens = line.split('\n')
    all_tokens = all_tokens[0]
    all_tokens = all_tokens.split(" ")

    ## Filtered tokens
    dict = {} # un dicionrio por linea
    s_token = []
    for index, token in enumerate(all_tokens):
        if index < len(all_tokens) and len(token) > L: # no consideres /n

            # Only Simple
            #token = wnl.stem(token)
            a = wnl.lemmatize(token)
            if a == token:
                token = wnl.lemmatize(token,'v')
            else:
                token = a

            if token not in dict.keys():
                dict[token]=1
            else:
                dict[token]=dict[token]+1
            s_token.append(token)

            """
            The following lines can be used to see if n-grams > 1 increase
            classification accuracy.
            Only double considers pairs "the_cat"
            Only triples adds triples "the_cat_went"

            # Only double
            if index+1 < len(all_tokens):
                token1 = all_tokens[index]
                token2 = all_tokens[index+1]

                a = wnl.lemmatize(token1)
                if a == token1:
                    token1 = wnl.lemmatize(token1,'v')
                else:
                    token1 = a

                a = wnl.lemmatize(token2)
                if a == token2:
                    token2 = wnl.lemmatize(token2,'v')
                else:
                    token2 = a

                token = token1 + '_' + token2

                if token not in dict.keys():
                    dict[token]=1
                else:
                    dict[token]=dict[token]+1

            # Only Triple
            if index+2 < len(all_tokens):
                token1 = all_tokens[index]
                token2 = all_tokens[index+1]
                token3 = all_tokens[index+2]

                a = wnl.lemmatize(token1)
                if a == token1:
                    token1 = wnl.lemmatize(token1,'v')
                else:
                    token1 = a

                a = wnl.lemmatize(token2)
                if a == token2:
                    token2 = wnl.lemmatize(token2,'v')
                else:
                    token2 = a

                a = wnl.lemmatize(token3)
                if a == token3:
                    token3 = wnl.lemmatize(token3,'v')
                else:
                    token3 = a

                token = token1 + '_' + token2 + '_' + token3

                if token not in dict.keys():
                    dict[token]=1
                else:
                    dict[token]=dict[token]+1
                """
    return dict, s_token

## Complete text
def filter_text(file,N,L):
    lista = []
    lista_s = []
    ## Clean the data correctly
    my_string_punct = string.punctuation
    #my_string_punct = []

    ## Eliminate stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    #stopwords = []

    ## Lemmantization
    wnl = nltk.WordNetLemmatizer()

    ## Stemmer
    #wnl = PorterStemmer()
    #wnl = LancasterStemmer()
    #wnl = SnowballStemmer("english")

    for index,info in enumerate(file.readlines()):
        if index > 0: ## first line of file are labels
            info = info.split('","')
            line = info[1]
            if index < N:
                dict = []
                dict,s_token = filter_line(line,L,wnl,my_string_punct,stopwords)
                lista.append(dict)
                lista_s.append(s_token)
    return lista, lista_s

    """
    This is the main Function that loads "train.csv".
    N is the number of lines consider for the analysis.
    L is the minimum length of a token allowed for the analysis.
    """
def dos_listas(N,L):
    file_train = io.open('train.csv','r',encoding='utf-8')
    lista =[]
    lista,lista_s = filter_text(file_train, N,L)
    dim = len(lista)
    return lista, lista_s, dim
