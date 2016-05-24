# -*- coding: utf-8 -*-
import logging
import os
import os.path
from util import timed
from search import Indexable
from search import SearchEngine

logger = logging.getLogger(__name__)

class Lyric(Indexable):
    """Class encapsulating a specific behavior of indexed songs.
        
        Args:
        iid (int): Identifier of indexable objects.
        title (str): Title of the song  .
        singer (str): singer of the song.
        lyric (str): Lyric of the song.
        metadata (str): Plain text with data to be indexed.
        
        Attributes:
        title (str): Title of the song
        singer (str): singer of the song
        lyric (str): Lyric of the song
        
        """
    
    def __init__(self, iid, title, singer, lyric):
        Indexable.__init__(self, iid, lyric)
        self.title = title
        self.singer = singer
        self.lyric = lyric

    
    def __repr__(self):
        return 'id: %s, title: %s, singer: %s' % \
            (self.iid, self.title, self.singer)

class SongInfo(Indexable):
    """Class encapsulating a specific behavior of indexed songs.

        Args:
        iid (int): Identifier of indexable objects.
        title (str): Title of the song  .
        singer (str): singer of the song.
        metadata (str): Plain text with data to be indexed.

        Attributes:
        title (str): Title of the song
        singer (str): singer of the song

        """

    def __init__(self,iid,title,singer,metadata):
        Indexable.__init__(self,iid,metadata)
        self.title = title
        self.singer = singer

    def  __repr__(self):
        return 'id: %s, title: %s,singer: %s'% \
               (self.iid, self.title, self.singer)

class lyricInventory(object):
    """Class representing a inventory of lyrics.

    Args:
      filename (str): File name containing lyric inventory data.

    Attributes:
      filename (str): File name containing lyric inventory data.
      engine (SearchEngine): Object responsible for indexing lyric inventory data.

    """
    _NO_RESULTS_MESSAGE = 'Sorry, no results.'

    def __init__(self, filename):
        self.filename = filename
        self.engine = SearchEngine()
        self.engine2 = SearchEngine()

    @timed
    def load_lyrics(self):
        """Load lyrics from a file name.

        This method leverages the iterable behavior of File objects
        that automatically uses buffered IO and memory management handling
        effectively large files.

        """
        logger.info('Loading lyrics from file...')
        iid =  1
        for parent,dirnames,fnames in os.walk(self.filename):  
                for fname in fnames:
                    fname2 = './lrc/' + fname
                    lyric = open(fname2).read().strip()
                    temp = fname.rstrip('.lrc').split('-')
                    if len(temp)<=1:
                        continue
                    singer = temp[0]
                    title = temp[1]
                    metadata = singer + ' ' + title

                    lyricobject = Lyric(iid, title, singer,lyric)
                    songobject  = SongInfo(iid,title,singer,metadata)
                    self.engine.add_object(lyricobject)
                    self.engine2.add_object(songobject)
                    iid+=1
            
        self.engine.start()
        self.engine2.start()

    @timed
    def search_lyrics(self, query, n_results=10):
        """Search lyrics according to provided query of terms.

        The query is executed against the indexed lyrics, and a list of lyrics
        compatible with the provided terms is return along with their tf-idf
        score.

        Args:
          query (str): Query string with one or more terms.
          n_results (int): Desired number of results.

        Returns:
          list of IndexableResult: List containing lyrics and their respective
            tf-idf scores.

        """
        result = ''
        if len(query) > 0:
            result = self.engine.search(query, n_results)

        if len(result) > 0:
            return '\n'.join([str(indexable) for indexable in result])
        return self._NO_RESULTS_MESSAGE

    def search_info(self, query, n_results=10):
        """Search song information according to provided query of terms.

        The query is executed against the indexed lyrics, and a list of lyrics
        compatible with the provided terms is return along with their tf-idf
        score.

        Args:
          query (str): Query string with one or more terms.
          n_results (int): Desired number of results.

        Returns:
          list of IndexableResult: List containing lyrics and their respective
            tf-idf scores.

        """
        result = ''
        if len(query) > 0:
            result = self.engine2.search(query, n_results)

        if len(result) > 0:
            return '\n'.join([str(indexable) for indexable in result])
        return self._NO_RESULTS_MESSAGE


    def lyrics_count(self):
        """
        Returns:
          int: Number of lyrics indexed.
        """
        return self.engine.count()