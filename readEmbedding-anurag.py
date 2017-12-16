import sys
import numpy
from sklearn.metrics.pairwise import cosine_similarity


# Storing the word embeddings produced by word2vec
with open("TREC678-merged-processed.txt.vec.vec", "r") as fp: 
#with open("Nepal-earthquake-tweetidText-out-doc2vec-tweetsOnly.txt", "r") as fp:
    embedding_vectors = {}
    for line in fp:
        line = line.strip() #To remove '\n'
        line = line.split(" ") #To tokenize into words
        word = line[0]
        vector = line[1:]
        vector = map(float, vector)
        embedding_vectors[word] = numpy.array(vector)

        
#print cosine_similarity([embedding_vectors["</s>"]],[embedding_vectors["nepal"]])[0][0]

# Storing the tweets
#with open("training-LabelDocidDocumentTerms.txt", "r") as tw: 
with open("temp.txt", "r") as tw: 
    docs_textLabelIds = {}
    for line in tw:
        line = line.strip()
        line = line.split(" ")
        rel_label = line[0]
        docid = line[1]
        doctext = line[2:]
        tuple1 = (rel_label, docid)
        docs_textLabelIds[tuple1] = doctext


#querytweet = tweets_textAndIds["592162799498854400"]

#sum_word = numpy.zeros(2000)

#for word in querytweet:
#    if word in embedding_vectors.keys():
#        sum_word += embedding_vectors[word]

for tuple2, txt in docs_textLabelIds.iteritems():
    sum_word1 = numpy.zeros(200)
    for word in txt:
        if word in embedding_vectors.keys():
            sum_word1 += embedding_vectors[word]
    print tuple2[0] + " " + tuple2[1] + " ",
    print sum_word1
    #print sum_word1
