# Lab 1 extended

The cosine similarities for Lab 1 were updated to reflect the new LSA (5000 most common words plus other words in Table 1 of RG65 along with bigram counts that include both the word preceding and the word following). The cosine similarity for the Word2Vec model from gensim is also reported as SMw.

Cosine Similarities:
S and SM1:  (0.32505759628217934, 0.008772462880312493)
S and SM1+:  (0.22721923312614276, 0.070976920136765348)
S and SM2_10:  (0.14346035811964897, 0.25808683501731222)
S and SM2_50:  (0.37449198674120587, 0.0022986358951080348)
S and SM2_100:  (0.25016690772436939, 0.046183167388349933)
S and SMw:  (0.76909957887502234, 1.1479623700761867e-13)

The Word2Vec model has a much higher Pearson correlation than any of the LSA models.


Next, an analogy test is done using analogies from http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt. The lists capital-common-countries, capital-world, currency, city-in-state, and family are considered to be semantic analogies while the remaining lists (gram1-adjective-to-adverb, gram2-opposite, etc.) are considered to be syntactic analogies. An analogy test was done using semantic analogy tuples using the LSA model (100 dimensions i.e. SM2_100) and using the Word2Vec model from gensim. An analogy test was then done using syntactic analogy tuples using the same two models. 

There were only 90 semantic analogy tuples from the original data which had all words contained within the most common 5030 (a requirement to use our LSA model). For a fair comparison, only these 90 were used for each model. To stay consistent with the semantic number and to finish analogy tests in a timely manner, only 90 of the ~2000 syntactic tuples were used to run the analogy test (the 90 instances were spaced out so that some analogy tuples from each of the semantic lists was included). It is also worth noting that since the original LSA model unfortunately converted all words to lower case, any words that had capital letters including cities, countries, etc. would not be selected as part of the set of analogy tuples.

The results are below:

LSA semantic analogy accuracy:  11 / 90
Word2Vec semantic analogy accuracy 79 / 90
LSA syntactic analogy accuracy:  2 / 90
Word2Vec syntactic analogy accuracy 50 / 90

The Word2Vec gensim model does much better than the LSA model on both semantic and syntactic analogies. Both LSA and the Word2Vec model perform better on semantic analogies than they do on syntactic analogies for this data set. Because the data sets are small, these results may be skewed. The results are also skewed because we limit our vocabulary to only 5030 words, and ensure that the answer word to our analogies are one of these 5030. Since the word2vec model was actually trained on hundreds of thousands or millions of words, the probability that the closest matching word out of a list of only 5030 is the one we were looking for is much higher than if it had to search the entire training set of words to find the closest word vector.
