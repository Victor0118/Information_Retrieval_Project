"""word search engine.

This module executes a simple search for words given a term-based query. The
query is capture from STDIN and executed on an indexed data structure
containing words' information. The result is displayed on STDOUT and outputs
the top 10 matches ordered by their tf*idf scores.

Example:
    $ python word_index.py --data './lrc'

    word_index Loading words...
    word Loading words from file...
    search Start search engine (Indexing | Ranking)...
    search Vocabulary assembled with terms count 230885
    search Starting tf computation...
    search Starting tf-idf computation...
    search Starting tf-idf norm computation...
    search Building index...
    search Function = load_words, Time = 406.88 sec
    word_index Done loading words, 5478 docs in index

    Enter a query, or hit enter to quit: burn
    util : Function = search_words, Time = 0.01 sec
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

import word

DEBUG = True

# Log initialization
log_level = logging.DEBUG if DEBUG else logging.INFO
log_format = '%(asctime)s - %(levelname)s - %(module)s : %(lineno)d - %(message)s'
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__)

CATALOG_FILENAME = 'Reuters'
sys.path.append('./'+CATALOG_FILENAME)

    
def execute_search(data_location):
    """Capture query from STDIN and display the result on STDOUT.

    The query of terms is executed against an indexed data structure
    containing words' information. If not result is found, an warning message
    will notify the user of such situation.
-[ ]
    Args:
      data_location (str): Location of the data file that will be indexed.

    """
    query = None
    repository = word.wordInventory(data_location)
    logger.info('Loading...')

    repository.load_words()
    docs_number = repository.words_count()
    logger.info('Done loading words, %d docs in index', docs_number)


    print '============================================================='
    
    

    # print search_results
    
    choice = 1
    while choice!=3:
        choice = raw_input('bool search(enter 1) or common search(enter 2) or exit(enter 3):')
        if choice == '':
             continue
        choice = int(choice)
        
        if choice == 1:
            synchoice=raw_input('Do you want to add sysnonym search?(y/n)')
            if(synchoice=='y' or synchoice=='Y'):
                SYSNONYM=True
                print "You have chosen sysnonym search"
            elif(synchoice=='n'or synchoice=='N'):
                SYSNONYM=False
                print "You have canceled sysnonym search"
            else:
                SYSNONYM=False
                print "input error. default: no sysnonym search"          
        elif choice == 2:
            synchoice=raw_input('Do you want to add sysnonym search?(y/n)')
            if(synchoice=='y' or synchoice=='Y'):
                SYSNONYM=True
                print "You have chosen sysnonym search"
            elif(synchoice=='n'or synchoice=='N'):
                SYSNONYM=False
                print "You have canceled sysnonym search"
            else:
                SYSNONYM=False
                print "input error. default: no sysnonym search"   
        elif choice == 3:
            exit()
        else:
            print "input error"
        print '============================================================='
        query = raw_input('Enter a query, or hit enter to quit: ')
        
        search_results = repository.search_words(query,choice = choice,SYSNONYM=SYSNONYM)
   


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-d', '--data',
                      dest='data',
                      help='Location of the data file that will be indexed',
                      default=CATALOG_FILENAME)

    options, args = parser.parse_args()
    execute_search(options.data)
