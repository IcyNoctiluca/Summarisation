''' Web Technology Coursework    '''
''' Term Frequency Inverse Document Frequency Algorithm for Summarisation     '''
''' Python 3.6.3                 '''


''' importing the pkgs & libs    '''
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import os
import re
import spacy
import sys


''' setup global vars    '''
TEST_FILE_DIR = 'Test Files'                        # the directory containing the documents
MAX_SUMMARY_LENGTH = int( sys.argv[1] )             # number of words for a summary
DOCUMENT_NUMBER = len( os.listdir(TEST_FILE_DIR) )  # number of documents in the corpus
nlp = spacy.load('en')                              # english tokenizer, tagger etc. from spaCy


''' represents a sentence with useful attributes        '''
class Sentence:

    def __init__(self, text, adjustedWeight):

        # content of sentence
        self.text = text

        # adjust weight for location, sentence length etc.
        self.adjustedWeight = adjustedWeight


''' represents a single document with useful attributes     '''
class Document:

    def __init__(self, originalText):

        # initalise with original text from document
        self.originalText = originalText

        # variable to hold the cleaned text
        self.cleanText = None

        # to hold Sentence objects in order in which they appear in the text
        self.sentenceArray = None

        # dictionary to hold frequencies of each word in the vocab
        self.tfidfWordWeights = {}


''' function to clean a raw text body - remove puncutation, stop words etc.        '''
def cleanseText(text):

    # remove new line chars & make everything lowercase
    text = text.replace('\n', '\r').replace('\r', ' ').lower()

    # remove all remaining non-letter characters
    text = re.sub('[^a-z ]', '', text)

    return text


''' return the idf of a given word          '''
def getIDF(word):

    global DOCUMENT_NUMBER
    global textAr

    # number of docs in which the word appears
    docFreq = np.sum([1 for text in textAr if word in cleanseText(text)])

    return np.log(DOCUMENT_NUMBER / docFreq)


''' returns a dictionary with the relative tf-idf weight of each word        '''
def getTFIDFWeights(cleanText):

    # dictionary to hold frequencies of each word in the vocab
    tfidfWeights = {}

    # for each word in the text
    for word in word_tokenize(cleanText):

        # increment the occurence of the word in the dictionary
        if word not in tfidfWeights:
            tfidfWeights[word] = 1.0
        else:
            tfidfWeights[word] += 1.0


    # normalising the occurences into percentages
    # and multiplying the TF by the IDF for each word
    for word in tfidfWeights:
        tfidfWeights[word] *= getIDF(word) / len(word_tokenize(cleanText))

    return tfidfWeights


''' returns an array of Sentence objects, with their weights calculated
    based upon structural and locational heuristic information              '''
def getRankedSentences(text, tfidfWordWeights):

    # array to hold sentence objects
    sentAr = []

    # number of sentences in document
    sentCount = len(sent_tokenize(text))

    # decay function to decrease sentence weight with
    # increasing distance from the begining of the article
    locationalDecay = lambda X: np.exp( -X / sentCount )

    # iterate through sentences in document text
    # position indictates where abouts in the document the sentence is
    for position, sent in enumerate(sent_tokenize(text)):

        cleanSent = cleanseText(sent)

        # weight based on the sentence's words
        wordWeight = np.sum(
            [tfidfWordWeights[word]
                for word in word_tokenize(cleanSent)
                    if word in tfidfWordWeights]
                        )

        # adjust weight according to sentence length
        weightPerWord = wordWeight / len(sent.split(' '))

        # adjust weight based on sentence's position in the document
        finalAdjustedWeight = weightPerWord * locationalDecay(position)

        # add sentence object to array
        sentAr.append(Sentence(sent, finalAdjustedWeight))

    return sentAr


''' sets up a Document object, assigning each sentence with an adjusted rank    '''
def prepareDocument(text):

    # setup a Document object with the text as a field
    d = Document(text)

    # preprocess the text and store this in the object for further use
    d.cleanText = cleanseText(text)

    # compute frequencies of each word in cleaned text
    # and set this as a field of the object for further use
    d.tfidfWordWeights = getTFIDFWeights(d.cleanText)

    # set document field with array of Sentence objects
    # each Sentence object has attributes needed for ranking
    d.sentenceArray = getRankedSentences(text, d.tfidfWordWeights)

    return d


''' finds the best sentence from all documents
    returns in the index of both the document
    and the sentence in that document               '''
def getBestSentence(docAr):

    # vars to remember best sentence number and document from which it came
    bestSentDocNo, bestSentSentNo = None, None

    # var to find best sentence in document collection
    bestSentWeight = 0

    # for each Document
    for docNo, doc in enumerate(docAr):

        # for each Sentence in the Document
        for sentNo, sent in enumerate(doc.sentenceArray):

            # if the weight of the sentence is the best so far...
            if sent.adjustedWeight > bestSentWeight:

                # record the position of the sentence
                bestSentDocNo = docNo
                bestSentSentNo = sentNo

                # and update the newest best weight
                bestSentWeight = sent.adjustedWeight


    return bestSentDocNo, bestSentSentNo


''' returns true if the given sentence is deemed too
    similar to something already in the summary         '''
def sentNotTooSimilar(SUMMARY, bestSent):

    # check bestSent against each sentence already in the summary
    for sent in sent_tokenize(SUMMARY):

        # if sentences are similar above 90% threshold
        if nlp(u"" + sent).similarity(nlp(u"" + bestSent)) >= 0.9:

            # sentence is too similar
            return False

    # sentence is fine
    return True



if __name__ == '__main__':

    # main summary string
    SUMMARY = str()

    # var to check if sentences may be added or if summary is at max length
    isSummaryFull = False

    # array to hold the text of each document
    textAr = np.array(
            [open(os.path.join(TEST_FILE_DIR, docName)).read()
                for docName in os.listdir(TEST_FILE_DIR)]
                    )

    # array to hold each Document object
    docAr = np.array(
            [prepareDocument(text)      # set up a new Document object for each text with the attributes computed
                for text in textAr]
                    )

    # while summary not at max length, add sentences to it
    while not(isSummaryFull):

        # gets indexing info of current best sentence
        bsDocNo, bsSentNo = getBestSentence(docAr)

        # best sentence
        bestSent = docAr[bsDocNo].sentenceArray[bsSentNo].text

        # check if summary is not at max length
        if len((SUMMARY + bestSent).split(' ')) <= MAX_SUMMARY_LENGTH:

            # check that sentence isn't too similar to something already in summary
            if sentNotTooSimilar(SUMMARY, bestSent):

                # add best sentence to summary
                SUMMARY += bestSent + ' '

        else:
            isSummaryFull = True

        # remove best sentence from document so it can recompute next best sentence
        del docAr[bsDocNo].sentenceArray[bsSentNo]

        # the original text without the just-found best sentence
        textWithoutSent = ' '.join([s.text for s in docAr[bsDocNo].sentenceArray])

        # replace the Document object without that sentence
        # and update all the weights of the remaining sentences
        docAr[bsDocNo] = prepareDocument(textWithoutSent)

    print (SUMMARY)
