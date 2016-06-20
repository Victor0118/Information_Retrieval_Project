# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import scipy.sparse.sparsetools as sptools
import logging
import synonym as sy
import re
from collections import defaultdict
from token import RIGHTSHIFTEQUAL
from Tkconstants import LEFT
from checkSpelling import checkSpelling
from boolParser import exprNode, parser
import main
logger = logging.getLogger(__name__)
originword=set()

STOP_WORDS_FILENAME = 'stop_words.txt'

DEBUG=main.DEBUG

class Indexable(object):
    """Class representing an object that can be indexed.

    It is a general abstraction for indexable objects and can be used in
    different contexts.

    Args:
      iid (int): Identifier of indexable objects.
      metadata (str): Plain text with data to be indexed.

    Attributes:
      iid (int): Identifier of indexable objects.
      words_count (dict): Dictionary containing the unique words from
        `metadata` and their frequency.

    """

    def __init__(self, iid, metadata):
        self.iid = iid
        self.words_count = defaultdict(int)

        # markFixWordList = sy.markDelete(metadata)

        # stemmedwordslist = sy.stemminglist(markFixWordList)

        # print stemmedwordslist

        lastword = ""
        metadatas = re.findall(r"[a-zA-Z]+", metadata.lower())
        # metadatas = sy.markDelete(metadatas)
        # print metadatas
        for word in metadatas:
            originword.add(word)
            word = sy.stemming(word)
            if lastword == "":
                lastword = word
                continue
            self.words_count[lastword + '_' + word] += 1
            lastword = word
        for word in metadatas:
            word = sy.stemming(word)
            self.words_count[word] += 1


    def __repr__(self):
        return '_'.join(self.words_count.keys()[:10])

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def words_generator(self, stop_words):
        """Yield unique words extracted from indexed metadata.

        Args:
          stop_words (list of str): List with words that mus be filtered.

        Yields:
          str: Unique words from indexed metadata.

        """
        # ret = []
        for word in self.words_count.keys():
            if word not in stop_words or len(word) > 5:
                # ret.append(word)
                yield word
        # return ret

    def count_for_word(self, word):
        """Frequency of a given word from indexed metadata.

        Args:
          word (str): Word whose the frequency will be retrieved.

        Returns:
          int: Number of occurrences of a given word.
bu
        """
        return self.words_count[word] if word in self.words_count else 0

class IndexableResult(object):
    """Class representing a search result with a tf-idf score.

    Args:
      score (float): tf-idf score for the result.
      indexable (Indexable): Indexed object.

    Attributes:
      score (float): tf-idf score for the result.
      indexable (Indexable): Indexed object.

    """

    def __init__(self, score, indexable):
        self.score = score
        self.indexable = indexable

    def __repr__(self):
        return 'score: %f, indexable: %s' % (self.score, self.indexable)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and abs(self.score - other.score) < 0.0001
                and self.indexable == other.indexable)

    def __ne__(self, other):
        return not self.__eq__(other)


class TfidfRank(object):
    """Class encapsulating tf-idf ranking logic.

    Tf-idf means stands for term-frequency times inverse document-frequency
    and it is a common term weighting scheme in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    Args:
      smoothing (int, optional): Smoothing parameter for tf-idf computation
        preventing by-zero divisions when a term does not occur in corpus.

      stop_words (list of str): Stop words that will be filtered during docs
        processing.

    Attributes:
      smoothing (int, optional): Smoothing parameter for tf-idf computation
        preventing by-zero divisions when a term does not occur in corpus.

      stop_words (list of str): Stop words that will be filtered during docs
        processing.

      vocabulary (dict): Dictionary containing unique words of the corpus as
        keys and their respective global index used in tf-idf data structures.

      ft_matrix (matrix): Matrix containing the frequency term for each term
        in the corpus respecting the index stored in `vocabulary`.

      ifd_diag_matrix (list): Vector containing the inverse document
        frequency for each term in the corpus. It respects the index stored in
        `vocabulary`.

      tf_idf_matrix (matrix): Matrix containing the ft-idf score for each term
        in the corpus respecting the index stored in `vocabulary`.

    """

    def __init__(self, stop_words, smoothing=1):
        self.smoothing = smoothing
        self.stop_words = stop_words
        self.vocabulary = {}
        # self.vocabulary_withoutsynword = {}
        self.ft_matrix = []
        self.ifd_diag_matrix = []
        self.tf_idf_matrix = []

    def build_rank(self, objects):
        """Build tf-idf ranking score for terms in the corpus.

        Note:
          The code in this method could have been extracted to other smaller
          methods, improving legibility. This extraction has not been done so
          that its runtime complexity can be computed easily (the runtime
          complexity can be improved).

        Args:
          objects (list of Indexable): List of indexed objects that will be
            considered during tf-idf score computation.

        """
        self.__build_vocabulary(objects)

        n_terms = len(self.vocabulary)
        n_docs = len(objects)
        ft_matrix = sp.lil_matrix((n_docs, n_terms), dtype=np.dtype(float))

        logger.info('Vocabulary assembled with terms count %s', n_terms)

        # compute idf
        logger.info('Starting tf computation...')
        for index, indexable in enumerate(objects):
            for word in indexable.words_generator(self.stop_words):
                word_index_in_vocabulary = self.vocabulary[word]
                doc_word_count = indexable.count_for_word(word)
                ft_matrix[index, word_index_in_vocabulary] = doc_word_count
        
        # Add synword's idf

        # for word in self.vocabulary_withoutsynword.keys():
        #     for synword in sy.synonymwords(word)[0:4]:
        #         word_index_in_vocabulary = self.vocabulary[word]
        #         synword_index_in_vocabulary = self.vocabulary[synword]
        #         if synword not in self.vocabulary_withoutsynword.keys():
        #             #print "origin word: ", word," synword: ",synword
        #             ft_matrix[:,synword_index_in_vocabulary] = ft_matrix[:,word_index_in_vocabulary]
        #         elif synword != word:
        #             newmatrix1 = 0.6*ft_matrix[:,word_index_in_vocabulary]+0.4*ft_matrix[:,synword_index_in_vocabulary]
        #             newmatrix2 = 0.4*ft_matrix[:,word_index_in_vocabulary]+0.6*ft_matrix[:,synword_index_in_vocabulary]
        #             ft_matrix[:,word_index_in_vocabulary] = newmatrix1
        #             ft_matrix[:,synword_index_in_vocabulary] = newmatrix2



        self.ft_matrix = ft_matrix.tocsc()

        logger.info('Starting tf-idf computation...')
        # compute idf with smoothing
        df = np.diff(self.ft_matrix.indptr) + self.smoothing
        n_docs_smooth = n_docs + self.smoothing

        # create diagonal matrix to be multiplied with ft
        idf = np.log(float(n_docs_smooth) / df) + 1.0
        self.ifd_diag_matrix = sp.spdiags(idf, diags=0, m=n_terms, n=n_terms)

        # compute tf-idf
        self.tf_idf_matrix = self.ft_matrix * self.ifd_diag_matrix
        self.tf_idf_matrix = self.tf_idf_matrix.tocsr()

        # compute td-idf normalization
        norm = self.tf_idf_matrix.tocsr(copy=True)
        norm.data **= 2
        norm = norm.sum(axis=1)
        n_nzeros = np.where(norm > 0)
        norm[n_nzeros] = 1.0 / np.sqrt(norm[n_nzeros])
        norm = np.array(norm).T[0]
        sptools.csr_scale_rows(self.tf_idf_matrix.shape[0],
                                      self.tf_idf_matrix.shape[1],
                                      self.tf_idf_matrix.indptr,
                                      self.tf_idf_matrix.indices,
                                      self.tf_idf_matrix.data, norm)

    def __build_vocabulary(self, objects):
        """Build vocabulary with indexable objects.

        Args:
          objects (list of Indexable): Indexed objects that will be
            considered during ranking.

        """
        vocabulary_index = 0
        for obj in objects:
            position = obj.iid
            word = obj.word
            lastw = ""
            for w in word.split():
                if lastw == "":
                    lastw = w
                    continue
                if lastw + '_' + w in self.vocabulary:
                    continue
                self.vocabulary[lastw + '_' + w] = vocabulary_index
                vocabulary_index += 1
                lastw = w

        for indexable in objects:
            for word in indexable.words_generator(self.stop_words):
                if word not in self.vocabulary:
                    self.vocabulary[word] = vocabulary_index
                    vocabulary_index += 1


        # Backup origin vocabulary
        # self.vocabulary_withoutsynword = self.vocabulary.copy()
        #print "before: 1:",len(self.vocabulary_withoutsynword)," 2: ",len(self.vocabulary)
        # Add synword if it is not in vocabulary
        # for word in self.vocabulary_withoutsynword.keys():
        #     for synword in sy.synonymwords(word)[0:4]:
        #         if synword not in self.vocabulary.keys():
        #             self.vocabulary[synword] = vocabulary_index
        #             print "word: ", word, "synword: ", synword, ", index: ", vocabulary_index
        #             vocabulary_index += 1


        #print "after: 1:",len(self.vocabulary_withoutsynword)," 2: ",len(self.vocabulary)




    def compute_rank(self, doc_index, terms):
        """Compute tf-idf score of an indexed document.

        Args:
          doc_index (int): Index of the document to be ranked.
          terms (list of str): List of query terms.

        Returns:
          float: tf-idf of document identified by its index.

        """
        score = 0
        for term in terms:
            if term in self.stop_words:
                continue
            if term not in self.vocabulary.keys():
                continue
            term_index = self.vocabulary[term]
            score += self.tf_idf_matrix[doc_index, term_index]
        return score


class Index(object):
    """Class responsible for indexing objects.

    Note:
      In case of a indexer object, we dropped the runtime complexity for a
      search by increasing the space complexity. It is the traditional
      trade-off and here we are more interested in a lightening fast
      search then saving some space. This logic may have to be revisited if the
      index become too large.

    Args:
      term_index (dict): Dictionary containing a term as key and a list of all
        the documents that contain that key/term as values
      stop_words (list of str): Stop words that will be filtered during docs
        processing.

    Attributes:
      term_index (dict): Dictionary containing a term as key and a list of all
        the documents that contain that key/term as values
      stop_words (list of str): Stop words that will be filtered during docs
        processing.

    """

    def __init__(self, stop_words):
        self.stop_words = stop_words
        self.term_index = defaultdict(list)

    def build_index(self, objects):
        """Build index the given indexable objects.

        Args:
          objects (list of Indexable): Indexed objects that will be
            considered during search.

        """

        for position, indexable in enumerate(objects):
            for word in indexable.words_generator(self.stop_words):
                # build dictionary where term is the key and an array
                # of the IDs of indexable object containing the term
                self.term_index[word].append(position)

        # sy.syndictionary = sy.buildSyn(self.term_index.keys())
        # sy.saveToFile()

        # for word in self.term_index.keys():
        #     for synword in sy.synonymwords(word)[0:4]:
        #         if synword not in self.term_index.keys():
        #             self.term_index[synword] = self.term_index[word]
        #         elif synword != word:
        #             list1 = self.term_index[synword]
        #             list2 = self.term_index[word]
        #             self.term_index[word].extend(list2)
        #             self.term_index[synword].extend(list1)
        #             newlist1 = []
        #             for iid in self.term_index[word]:
        #                 if iid not in newlist1:
        #                     newlist1.append(iid)
        #             self.term_index[word] = newlist1

        #             newlist2 = []
        #             for iid in self.term_index[synword]:
        #                 if iid not in newlist2:
        #                     newlist2.append(iid)
        #             self.term_index[synword] = newlist2




    def search_terms(self, terms):
        """Search for terms in indexed documents.

        Args:
          terms (list of str): List of terms considered during the search.

        Returns:
          list of int: List containing the index of indexed objects that
            contains the query terms.

        """
        docs_indices = []
        num_stop_word = 0
        for term_index, term in enumerate(terms):
            # keep only docs that contains all terms
            if term in self.stop_words:
                num_stop_word += 1
                continue

            if term not in self.term_index:
                docs_indices = []
                break

            # compute intersection between results
            # there is room for improvements in this part of the code
            docs_with_term = self.term_index[term]
            #print docs_with_term
            if term_index-num_stop_word == 0:
                docs_indices = docs_with_term
            else:
                docs_indices = set(docs_indices) & set(docs_with_term)

        if(DEBUG):
            print "debug info in search_terms",term,":",list(docs_indices)

        return list(docs_indices)


class SearchEngine(object):
    """Search engine for objects that can be indexed.

    Attributes:
      objects (list of Indexable): List of objects that can be considered
        during search.
      stop_words (list of str): Stop words that will be filtered during docs
        processing.
      rank (TfidfRank): Object responsible for tf-idf ranking computation.
      index (Index): Object responsible for data indexing.

    """

    def __init__(self):
        self.objects = []
        self.stop_words = self.__load_stop_words()
        self.rank = TfidfRank(self.stop_words)
        self.index = Index(self.stop_words)

    def __load_stop_words(self):
        """Load stop words that will be filtered during docs processing.
        Returns:
          list str: List of English stop words.
        """
        stop_words = {}
        with open(STOP_WORDS_FILENAME) as stop_words_file:
            for word in stop_words_file:
                stop_words[word.strip()] = True
        return stop_words

    def add_object(self, indexable):
        """Add object to index.
        Args:
          indexable (Indexable): Object to be added to index.
        """
        # print indexable
        self.objects.append(indexable)

    def start(self):
        """Perform search engine initialization.

        The current implementation initialize the ranking and indexing of
        added objects. The code below is not very efficient as it iterates over
        all indexed objects twice, but can be improved easily with generators.

        """
        logger.info('Start search engine (Indexing | Ranking)...')
        self.index.build_index(self.objects)
        self.rank.build_rank(self.objects)

    def search(self, query, n_results=10,SYSNONYM=False):
        """Return indexed documents given a query of terms.

        Assumptions:
          1) We assume all terms in the provided query have to be found.
          Otherwise, an empty list will be returned. It is a simple
          assumption that can be easily changed to consider any term.

          2) We do not use positional information of the query term. It is
          not difficult whatsoever to take it into account, but it was just a
          design choice since this requirement was not specified.

        Args:
          query (str): String containing one or more terms.
          n_results (int): Desired number of results.

        Returns:
          list of IndexableResult: List of search results including the indexed
            object and its respective tf-idf score.

        """
        terms = query.lower().split()
        originterms=terms
        dictionary = self.index.term_index.keys()
        global originword


        
        


        
        termslist=[]
        termslist.append(terms)        
        if(SYSNONYM):       
            #add sysnon termlist
            if(DEBUG):
                print 'add sysnonym'
            for t_index,t in enumerate(terms):
                if t!='and' and t!='or'and t!='not'and t!='('and t!=')':
                    sysnlist=sy.synonymwords(t)
                    if(DEBUG):
                        print "sysnlist",sysnlist
                    if len(sysnlist)!=0:
                        for sysn in sysnlist:
                            if sysn!=t:
                                sysnterms=[]+terms                       
                                #print "systerms",sysnterms,"terms",terms, "....sysn:",sysn
                                del(sysnterms[t_index])
                                sysnterms.insert(t_index,sysn)
                                termslist.append(sysnterms)


        #merge docset
        docset=set()
        for terms in termslist:
            #add stemming
            terms = sy.stemminglist(terms)
            #add multiple words
            newterms = []
            if len(terms) > 1:
                lastt = ""
                for t in terms:
                    #checkSpelling(t, originword)
                    if lastt == "":
                        lastt = t
                        continue
                    newterms.append(lastt+'_'+t)
                    lastt = t
                terms = newterms
            docs_indices = self.index.search_terms(terms)
            docset=docset|set(docs_indices)


        #if too few documents returned, add original words
        if len(docset)<20:
            for terms in termslist:
                #add stemming
                terms = sy.stemminglist(terms)
                docs_indices = self.index.search_terms(terms)
                docset=docset|set(docs_indices)

        #if too few documents returned, add single word
        if len(docset)<=10:
            for terms in termslist:
                #add stemming
                terms = sy.stemminglist(terms)
                for t in terms:
                    docs_indices= self.index.search_terms(t.split())
                    docset=docset|set(docs_indices)


        #calculate scores
        search_results = []
        for doc_index in list(docset):
            indexable = self.objects[doc_index]
            doc_score = self.rank.compute_rank(doc_index, terms)
            result = IndexableResult(doc_score, indexable)
            search_results.append(result)
        search_results.sort(key=lambda x: x.score, reverse=True)

        #too few results: checkspelling
        if(len(docset)<3):
            for t in originterms:
                if(DEBUG):
                    print  "check :",t
                checkSpelling(t, originword)
        return search_results[:n_results]


    def search_bool(self,query,n_results=10,SYSNONYM=False):
        """return all documents that satisfy the demand given a bool expression
        Assumptions:
            1)we take 'and'_'or'_'not' as conjunction word
        Args:
          query (str): String of bool expression.
          n_results (int): Desired number of results.

        Returns:
          list of  docs indexes
        """
        terms=query.lower().split()
       
        global originword
        dictionary = self.index.term_index.keys()
        #dealing with branket
        for term_index,term in enumerate(terms):
            branket=''
            if (term!='(' and term!=')'):
                a=term.find('(')
                b=term.find(')')
                if(a!=-1 and b==-1):
                    branket='('
                elif(a==-1 and b!=-1):
                    branket=')'
                elif (a!=-1 and b!=-1 and a<b):
                    branket='('
                elif (a!=-1 and b!=-1 and a>b):
                    branket=')'
            if branket!='':
                c_index=term.find(branket)
                #print c_index
                terms.remove(term)
                if c_index==0:
                    terms.insert(term_index,branket)
                    terms.insert(term_index+1,term.strip().lstrip(branket).rstrip(branket))
                elif c_index==len(term)-1:
                    terms.insert(term_index,term.strip().lstrip(branket).rstrip(branket))
                    terms.insert(term_index+1,branket)
                elif c_index!=-1:
                    tt=term.split(branket)
                    terms.insert(term_index,tt[0])
                    terms.insert(term_index+1,branket)
                    terms.insert(term_index+2,tt[1])
       
        originterms=terms
        
       
        
        termslist=[]
        termslist.append(terms)        
        if(SYSNONYM):      
            #add sysnon termlist
            for t_index,t in enumerate(terms):
                if t!='and' and t!='or'and t!='not'and t!='('and t!=')':
                    sysnlist=sy.synonymwords(t)
                    if len(sysnlist)>0:
                        for sysn in sysnlist:
                            if sysn!=t:
                                sysnterms=[]+terms
                                del(sysnterms[t_index])
                                sysnterms.insert(t_index,sysn)
                                if(DEBUG):
                                    print "new systerms",sysnterms, "....sysn:",sysn
                                termslist.append(sysnterms)
       

        #merge docset 
        docset=set()
        for terms in termslist:
            # add stemming
            terms = sy.stemminglist(terms)
            #add multiple words
            newterms = []
            if len(terms) > 1:
                nextt = ""
                cnt=0
                first=1              
                for t_index,t in enumerate(terms):

                    if t_index+1 < len(terms):
                        nextt = terms[t_index+1]
                    else:
                        nextt=""
                    if t!='and' and t!='or'and t!='not'and t!='('and t!=')':
                        if nextt!='and' and nextt!='or'and nextt!='not'and nextt!='('and nextt!=')' and nextt!="":
                            if first==0:
                                newterms.append('and')
                            newterms.append(t+'_'+nextt)
                            first=0
                        else:
                            if(first!=0):
                                newterms.append(t)
                            first=1                                               
                    else:
                        newterms.append(t)
                
                terms=newterms
                    #print t_index,terms[t_index]

            #else:
                # print  "check :",t
                # checkSpelling(terms[0], originword)
            if(DEBUG):
                print "debug info in search_bool:",terms
        
            expRoot=parser(terms).parse()
            #docs_indiex = self.index.search_terms(term)
            search_results = []
            if expRoot!=None:
                #expRoot.printout()
                docs_indices = expRoot.calc(self.index,self.count())
                docset=set(docs_indices)|docset

        #too few results: checkspelling
        if(len(docset)<3):
            for t in originterms:
                if t!='and' and t!='or'and t!='not'and t!='('and t!=')':
                    if(DEBUG):
                        print  "check :",t
                    checkSpelling(t, originword)

        return list(docset)[:n_results]



    def count(self):
        """Return number of objects a
        eady in the index.
        Returns:
          int: Number of documents indexed.
        """
        return len(self.objects)
