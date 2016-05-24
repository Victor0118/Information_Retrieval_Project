"""lyric search engine.

This module executes a simple search for lyrics given a term-based query. The
query is capture from STDIN and executed on an indexed data structure
containing lyrics' information. The result is displayed on STDOUT and outputs
the top 10 matches ordered by their tf*idf scores.

Example:
    $ python lyric_index.py --data './lrc'

    lyric_index Loading lyrics...
    lyric Loading lyrics from file...
    search Start search engine (Indexing | Ranking)...
    search Vocabulary assembled with terms count 230885
    search Starting tf computation...
    search Starting tf-idf computation...
    search Starting tf-idf norm computation...
    search Building index...
    search Function = load_lyrics, Time = 406.88 sec
    lyric_index Done loading lyrics, 5478 docs in index

    Enter a query, or hit enter to quit: burn
    util : Function = search_lyrics, Time = 0.01 sec
    score: 0.553258, indexable: id: 5255, title: Burn, singer: Usher
    score: 0.230353, indexable: id: 1614, title: Still Breathing, singer: duran duran
    score: 0.206084, indexable: id: 2882, title: Who We Are, singer: Lifehouse
    score: 0.201529, indexable: id: 5447, title: Lover Won't You Stay, singer: Will Young
    score: 0.174895, indexable: id: 3622, title: I Do It For You, singer: Nick Lachey
    score: 0.172824, indexable: id: 538, title: Wildsurf, singer: Ash
    score: 0.172410, indexable: id: 626, title: Fue, singer: Avril Lavigne
    score: 0.167547, indexable: id: 5003, title: Time To Burn, singer: The Rasmus
    score: 0.160747, indexable: id: 4076, title: Feve, singer: Ray Charles
    score: 0.159545, indexable: id: 4064, title: I Melt, singer: Rascal Flatts
    ...

"""
import sys
import optparse
import logging
sys.path.append('./lrc')
import lyric

DEBUG = True

# Log initialization
log_level = logging.DEBUG if DEBUG else logging.INFO
log_format = '%(asctime)s - %(levelname)s - %(module)s : %(lineno)d - %(message)s'
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__)

CATALOG_FILENAME = './lrc'


def execute_search(data_location):
    """Capture query from STDIN and display the result on STDOUT.

    The query of terms is executed against an indexed data structure
    containing lyrics' information. If not result is found, an warning message
    will notify the user of such situation.
-[ ]
    Args:
      data_location (str): Location of the data file that will be indexed.

    """
    query = None
    repository = lyric.lyricInventory(data_location)
    logger.info('Loading...')

    repository.load_lyrics()
    docs_number = repository.lyrics_count()
    logger.info('Done loading lyrics, %d docs in index', docs_number)

    choice = raw_input('Please choose to search lyrics(enter 1) or song info(enter 2) :')

    if choice == '1':
        while query is not '':
            query = raw_input('Enter a query, or hit enter to quit: ')
            search_results = repository.search_lyrics(query)

            print search_results

    elif choice == '2':
        while query is not '':
            query = raw_input('Enter a query, or hit enter to quit: ')
            search_results = repository.search_info(query)

            print search_results

    else:
        print 'choice error'
        exit()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-d', '--data',
                      dest='data',
                      help='Location of the data file that will be indexed',
                      default=CATALOG_FILENAME)

    options, args = parser.parse_args()
    execute_search(options.data)
