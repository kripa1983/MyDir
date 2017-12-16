#cat output1.txt | tr "\n" " "|sed 's/\[//g'|sed 's/\]//g'|tr -s " " > output2.txt

import sys
import numpy
from sklearn.metrics.pairwise import cosine_similarity


# Storing the word embeddings produced by word2vec
with open("output2.txt", "r") as fp: 
#with open("Nepal-earthquake-tweetidText-out-doc2vec-tweetsOnly.txt", "r") as fp:
    embedding_vectors = {}
    for line in fp:
        line = line.strip() #To remove '\n'
        line = line.split(" ") #To tokenize into words
        label = line[0]
        docid = line[1]
        tuple = (label, docid)
        vector = line[2:]
        #vector = line[3:-1]
        
        vector = map(float, vector)
        embedding_vectors[tuple] = numpy.array(vector)

        
#print cosine_similarity([embedding_vectors["</s>"]],[embedding_vectors["nepal"]])[0][0]


for tuple2, txt in embedding_vectors.iteritems():
    
    print tuple2[0] + " " + tuple2[1] + " ",
    print txt
    #print sum_word1
