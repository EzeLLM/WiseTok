# Author: Allan Caminha Trevisan <allan.trvsn@gmail.com>
# (c) 2014
#
# License: MIT

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

from nltk import stem

from core.textutils.text_pre_processing import get_stopwords_list

from time import time
import os
import pickle

STOPWORDS = get_stopwords_list(os.getcwd() + r"\core\textutils\stopwords.txt")

class FeatureExtractor:
    """This class encapsulate the functionalities from sklearn vectorizers.
    
    It uses one of the sklearn.feature_extraction.text vectorizers (count, tfidf, hash, ...)
    to convert a collection of text documents to a matrix of features. 

    Parameters
    -----------
    raw_data : list
        A list of documents, each document is represented by a string.

    ngram_range : tuple (default (1, 1))
        The lower and upper boundary of the range of n-values for different n-grams to be extracted.

    max_df : float in range [0.0, 1.0] or int, optional, (default 0.5)
        When building the vocabulary ignore terms that have a document frequency strictly higher than 
        the given threshold (corpus-specific stop words). If float, the parameter represents a proportion 
        of documents, integer absolute counts.

    min_df :  float in range [0.0, 1.0] or int, optional, (default 2)
        When building the vocabulary ignore terms that have a document frequency strictly lower than the
        given threshold. This value is also called cut-off in the literature. If float, the parameter represents 
        a proportion of documents, integer absolute counts.

    max_features : int (default None)
        If not None, build a vocabulary that only consider the top max_features ordered by term frequency across 
        the corpus.

    use_stemming : Boolean (default False)
            Perform sufix elimination to reduce data dimensionality.

    binary : Boolean (default False)
            If True, all non-zero term counts are set to 1. 
            This does not mean outputs will have only 0/1 values, only that 
            the tf term (in case of a tfidf vectorizer) in tf-idf is binary. 
            (Set idf to False to get 0/1 outputs.)

    remove_stopwords : Boolean (default True)
            Eliminate non informative words from dataset.

    Atributes
    ----------
    training_data : array[n_samples, n_features]
        Vectorized trainning data.

    vectorizer : sklearn.text vectorizer object
        Reference to the vectorizer object that handles feature
        extraction.

    feature_names : list of str
        The  feature index in the list maps right to the 
        respective column in X.

    vocabulary : dict mapping str to int
        Map the feature name to the respective column in X.

    tokenize : callable
        Handles preprocessing and tokenization so it is possible
        to recover the feature names present in a given tweet.
    """

    def __init__(self, raw_data, ngram_range=(1,1), max_df=0.5, min_df=2, max_features=None,
                 use_stemming=False, binary=False, remove_stopwords=True):
        
        self.raw_data = raw_data

        self.ngram_range = ngram_range

        if remove_stopwords:
            self.stop_words = STOPWORDS
        else:
            self.stop_words = []

        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features

        self.binary = binary

        if use_stemming:
            
            #print("Performing stemming...")
            t0 = time()
            stemmer = stem.RSLPStemmer()

            stemmed_tweets = []
            for tweet in self.raw_data:
                
                tweet_words = tweet.split()
                stemmed_tweet = []
                for word in tweet_words:

                    stemmed_word = stemmer.stem(word)
                    stemmed_tweet.append(stemmed_word)
                stemmed_tweets.append(" ".join(stemmed_tweet))
            
            #print("done in %fs" % (time() - t0))
            self.raw_data = stemmed_tweets

    def count_vectorizer(self):
        """Extract features using a count vectorizer."""

        #print("Extracting features from dataset using a count vectorizer")
        #print()

        #t0 = time()
        self.vectorizer = CountVectorizer(ngram_range=self.ngram_range, stop_words=self.stop_words, 
                                          max_df=self.max_df, min_df=self.min_df, max_features=self.max_features,
                                          binary=self.binary) 

        self.training_data = self.vectorizer.fit_transform(self.raw_data)
        self.vocabulary = self.vectorizer.vocabulary_
        self.feature_names = self.vectorizer.get_feature_names()

        #print("done in %fs" % (time() - t0))
        #print("n_samples: %d, n_features: %d" % self.training_data.shape)
        #print()

    def tfidf_vectorizer(self, use_idf=True):
        """Extract features using a tf-idf vectorizer.

        Parameters
        ----------
        use_idf : Boolean (default True)
            Perform inverse document frequency normalization
        """

        #print("Extracting features from dataset using a tf-idf vectorizer")
        #print()

        #t0 = time()
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, stop_words=self.stop_words, 
                                          max_df=self.max_df, min_df=self.min_df, max_features=self.max_features,
                                          use_idf=use_idf, binary=self.binary) 

        self.training_data = self.vectorizer.fit_transform(self.raw_data)
        
        self.vocabulary = self.vectorizer.vocabulary_
        self.feature_names = self.vectorizer.get_feature_names()
        
        #print("done in %fs" % (time() - t0))
        #print("n_samples: %d, n_features: %d" % self.training_data.shape)
        #print()

    def get_top_ngrams(self, max_ngrams=20):
        """Get the top ngrams extracted from some dataset.

        Calculate the ngrams weight based on the sum of
        the respective tfidf or count value in the respective
        column in the document term matrix.
        
        Parameters
        ----------        
        max_ngrams : int
            Max number of top ngrams to be retrived. 
            
        Returns
        -------
        top_ngrams : list of tuples (string, number)
            The global importance of each ngram in a text corpus in descending order. 
        """

        freqs = [(ngram, self.training_data.getcol(idx).sum()) for ngram, idx in self.vocabulary.items()]
        
        #sort from largest to smallest
        top_ngrams = sorted (freqs, key = lambda x: -x[1])

        if len(top_ngrams) >= max_ngrams:
            #for i, (ngram, weight) in enumerate(top_ngrams[:max_ngrams]):
            #    print ("%d: ngram: %s, weight: %s" % (i, ngram, weight))
            return top_ngrams[:max_ngrams]
        else:
            return top_ngrams

    def get_top_documents_by_cosine_similarity(self, max_documents=10):
        """Get the top most similar documents to the first document (our search querry).
        
        Parameters
        ----------
        max_documents : int
            The total number of documents to be retrived (the first one is
            the querry, so it is ignored later).
        """

        cosine_similarities = cosine_similarity(self.training_data[0:1], self.training_data)
        related_docs_indices = cosine_similarities.argsort().tolist()[0]
        related_docs_indices = related_docs_indices[:-max_documents:-1]
  
        top_documents = []
        for i in range(len(related_docs_indices)):
            top_documents.append(self.raw_data[related_docs_indices[i]])
 
        return top_documents[1:]

