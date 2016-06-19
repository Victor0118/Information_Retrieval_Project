from util import timed
import time
import logging
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet as wn
import pickle

logger = logging.getLogger(__name__)

syndictionary = {}

def stemming(word):
	st = LancasterStemmer()
	wordstem = st.stem(word)
	return wordstem

def stemminglist(wordlist):
	wordliststem = list()
	for word in wordlist:
		wordstem = stemming(word)
		wordliststem.append(wordstem)
	return wordliststem

def markDelete(wordsString):
	resultwordlist = list()
	for word in wordsString.split():
		fixword = word.replace(',','')
		# fixword = fixword.replace('.','')
    		if fixword[-1] == '.':
    	 		fixword = fixword[:-1]
		resultwordlist.append(fixword)

	return resultwordlist

# def saveToFile():
# 	syn_file = open('syn_file.txt', 'wb')
# 	pickle.dump(syndictionary, syn_file)
# 	syn_file.close()
# 	return 

# def loadFile():
# 	syndic = {}
# 	syn_file = open('syn_file.txt','rb')
# 	syndic = pickle.load(syn_file)
# 	syn_file.close()
# 	syndictionary = syndic
# 	return syndic

def searchSyn(word):
	synonymwordslist = list()
	synsetslist = wn.synsets(word)
	for synset in synsetslist:
		synonymwordslist.extend(synset.lemma_names())

	wordlist = []
	for word in synonymwordslist:
		if word not in wordlist:
			wordlist.append(word)

	return wordlist

# def buildSyn(wordlist):
# 	syndictionary = {}
# 	for originword in wordlist:
# 		synonymwordslist = list()
# 		synsetslist = wn.synsets(originword)
# 		for synset in synsetslist:
# 			synonymwordslist.extend(synset.lemma_names())

# 		synwordlist = []
# 		for word in synonymwordslist:
# 			if word not in synwordlist:
# 				synwordlist.append(word)

# 		# print "what is word:",originword
# 		syndictionary[originword] = synwordlist

# 	return syndictionary





# def synonymwords(word):
# 	if syndictionary.has_key(word):
# 		return syndictionary[word]
# 	else:
# 		searchSyn()

def synonymwords(word):
	return searchSyn(word)[0:3]

# syndictionary = loadFile()
