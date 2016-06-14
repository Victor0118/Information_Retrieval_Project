from util import timed
import time
import logging
logger = logging.getLogger(__name__)

def editDistanceCalc(query, word):
	l1 = len(query)
	l2 = len(word)
	distMat = [[0 for x in range(l2+1)] for y in range(l1+1)] 
	for i in range(l1+1):
		distMat[i][0] = i
	for j in range(l2+1):
		distMat[0][j] = j
	for i in range(1, l1+1):
		for j in range(1, l2+1):
			distMat[i][j] = min(distMat[i-1][j-1] + (0 if query[i-1]==word[j-1] else 1),
				distMat[i-1][j]+1, distMat[i][j-1]+1)
	return distMat[l1][l2]

@timed
def checkSpelling(query, dictionary):
	if query not in dictionary:
		minDist = len(query)+10
		for word in dictionary:
			if word[0]==query[0]:
				dist = editDistanceCalc(query, word)
				if dist < minDist:
					minDist = dist
					correctedWord = word
		print 'Did you mean '+correctedWord+' ?'

