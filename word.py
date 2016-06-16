# -*- coding: utf-8 -*-
import logging
import os
import os.path
from util import timed
from search import Indexable
from search import SearchEngine
from checkSpelling import checkSpelling
import pickle

logger = logging.getLogger(__name__)

class Word(Indexable):
    """Class encapsulating a specific behavior of indexed songs.

        Args:
        iid (int): Identifier of indexable objects.
        title (str): Title of the song  .
        singer (str): singer of the song.
        word (str): Word of the song.
        metadata (str): Plain text with data to be indexed.

        Attributes:
        title (str): Title of the song
        singer (str): singer of the song
        word (str): Word of the song

        """

    # def __init__(self, iid, title, singer, word):
    def __init__(self, iid, word):
        Indexable.__init__(self, iid, word)
        # self.title = title
        # self.singer = singer
        self.word = word


    # def __repr__(self):
    #     return 'id: %s, title: %s, singer: %s' % \
    #         (self.iid, self.title, self.singer)

# class SongInfo(Indexable):
#     """Class encapsulating a specific behavior of indexed songs.

#         Args:
#         iid (int): Identifier of indexable objects.
#         title (str): Title of the song  .
#         singer (str): singer of the song.
#         metadata (str): Plain text with data to be indexed.

#         Attributes:
#         title (str): Title of the song
#         singer (str): singer of the song

#         """

#     def __init__(self,iid,title,singer,metadata):
#         Indexable.__init__(self,iid,metadata)
#         self.title = title
#         self.singer = singer

#     def  __repr__(self):
#         return 'id: %s, title: %s,singer: %s'% \
#                (self.iid, self.title, self.singer)

class wordInventory(object):
    """Class representing a inventory of words.

    Args:
      filename (str): File name containing word inventory data.

    Attributes:
      filename (str): File name containing word inventory data.
      engine (SearchEngine): Object responsible for indexing word inventory data.

    """
    _NO_RESULTS_MESSAGE = 'Sorry, no results.'

    def __init__(self, filename):
        self.filename = filename
        self.engine = SearchEngine()
        self.engine2 = SearchEngine()

    @timed
    def load_words(self):
        """Load words from a file name.

        This method leverages the iterable behavior of File objects
        that automatically uses buffered IO and memory management handling
        effectively large files.

        """
        logger.info('Loading words from file...')
        iid =  1
        for parent,dirnames,fnames in os.walk(self.filename):
                for fname in fnames:
                    fname2 = './Reuters/' + fname
                    # print fname
                    word = open(fname2).read()
                    # temp = fname.rstrip('.html').split('-')
                    # if len(temp)<=1:
                        # continue
                    # singer = temp[0]
                    # title = temp[1]
                    # metadata = singer + ' ' + title

                    # wordobject = Word(iid, title, singer,word)
                    wordobject = Word(iid, word)
                    # songobject  = SongInfo(iid,title,singer,metadata)
                    self.engine.add_object(wordobject)
                    # self.engine2.add_object(songobject)
                    iid+=1

        self.engine.start()
        # self.engine2.start()
        self.saveToFile()

    @timed
    def search_words(self, query, n_results=10):
        """Search words according to provided query of terms.

        The query is executed against the indexed words, and a list of words
        compatible with the provided terms is return along with their tf-idf
        score.

        Args:
          query (str): Query string with one or more terms.
          n_results (int): Desired number of results.

        Returns:
          list of IndexableResult: List containing words and their respective
            tf-idf scores.

        """
        result = ''
        # dictionary = self.engine.index.term_index.keys()
        if len(query) > 0:
            # checkSpelling(query, dictionary)
            result = self.engine.search(query, n_results)
            print result

        if len(result) > 0:
            # return '\n'.join([str(indexable) for indexable in result])
            return
        return self._NO_RESULTS_MESSAGE

    # def search_info(self, query, n_results=10):
    #     """Search song information according to provided query of terms.

    #     The query is executed against the indexed words, and a list of words
    #     compatible with the provided terms is return along with their tf-idf
    #     score.

    #     Args:
    #       query (str): Query string with one or more terms.
    #       n_results (int): Desired number of results.

    #     Returns:
    #       list of IndexableResult: List containing words and their respective
    #         tf-idf scores.

    #     """
    #     result = ''
    #     if len(query) > 0:
    #         result = self.engine2.search(query, n_results)

    #     if len(result) > 0:
    #         return '\n'.join([str(indexable) for indexable in result])
    #     return self._NO_RESULTS_MESSAGE


    def saveToFile(self):
        fileObject = open('test.engine','w')
        pickle.dump(self.engine, fileObject)



    def words_count(self):
        """
        Returns:
          int: Number of words indexed.
        """
        return self.engine.count()
