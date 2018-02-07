import math
import operator
import random
import re
import numpy as np

#STEP 1
import nltk
from nltk.corpus import brown

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


#STEP 6. Table RG65 with lists P and S as defined in the lab.
P = [
('cord',	'smile'),
('rooster',	'voyage'),
('noon',	'string'),
('fruit',	'furnace'),
('autograph',	'shore'),
('automobile',	'wizard'),
('mound',	'stove'),
('grin',	'implement'),
('asylum',	'fruit'),
('asylum',	'monk'),
('graveyard',	'madhouse'),
('glass',	'magician'),
('boy',	'rooster'),
('cushion',	'jewel'),
('monk',	'slave'),
('asylum',	'cemetery'),
('coast',	'forest'),
('grin',	'lad'),
('shore',	'woodland'),
('monk',	'oracle'),
('boy',	'sage'),
('automobile',	'cushion'),
('mound',	'shore'),
('lad',	'wizard'),
('forest',	'graveyard'),
('food',	'rooster'),
('cemetery',	'woodland'),
('shore',	'voyage'),
('bird',	'woodland'),
('coast',	'hill'),
('furnace',	'implement'),
('crane',	'rooster'),
('hill',	'woodland'),
('car',	'journey'),
('cemetery',	'mound'),
('glass',	'jewel'),
('magician',	'oracle'),
('crane',	'implement'),
('brother',	'lad'),
('sage',	'wizard'),
('oracle',	'sage'),
('bird',	'crane'),
('bird',	'cock'),
('food',	'fruit'),
('brother',	'monk'),
('asylum',	'madhouse'),
('furnace',	'stove'),
('magician',	'wizard'),
('hill',	'mound'),
('cord',	'string'),
('glass',	'tumbler'),
('grin',	'smile'),
('serf',	'slave'),
('journey',	'voyage'),
('autograph',	'signature'),
('coast',	'shore'),
('forest',	'woodland'),
('implement',	'tool'),
('cock',	'rooster'),
('boy',	'lad'),
('cushion',	'pillow'),
('cemetery',	'graveyard'),
('automobile',	'car'),
('midday',	'noon'),
('gem',	'jewel')
]

S = [0.02,
     0.04,
     0.04,
     0.05,
     0.06,
     0.11,
     0.14,
     0.18,
     0.19,
     0.39,
     0.42,
     0.44,
     0.44,
     0.45,
     0.57,
     0.79,
     0.85,
     0.88,
     0.9,
     0.91,
     0.96,
     0.97,
     0.97,
     0.99,
     1,
     1.09,
     1.18,
     1.22,
     1.24,
     1.26,
     1.37,
     1.41,
     1.48,
     1.55,
     1.69,
     1.78,
     1.82,
     2.37,
     2.41,
     2.46,
     2.61,
     2.63,
     2.63,
     2.69,
     2.74,
     3.04,
     3.11,
     3.21,
     3.29,
     3.41,
     3.45,
     3.46,
     3.46,
     3.58,
     3.59,
     3.6,
     3.65,
     3.66,
     3.68,
     3.82,
     3.84,
     3.88,
     3.92,
     3.94,
     3.94]


#STEP 2 5030 most frequent words stored using uni and unigram
#unigram['example'] returns the index of the tuple ('example', count) in
#the uni list, where count is the unigram count of 'example'.
unigram = dict()
bigram = dict()
words = brown.words()
for w in words:
	w = re.sub('[^A-Za-z]+', '', w)
	if w != '':
		if not w.lower() in unigram:
			unigram[w.lower()] = 1
		else:
			unigram[w.lower()] += 1


old = unigram
uni = sorted(unigram.items(), key=operator.itemgetter(1), reverse=True)
sum = 0.0
for i,j in uni:
    sum += j

uni = uni[:5030]
for i in range(5030):
	unigram[uni[i][0]] = i


#The below chunk of code makes sure all words from table RG65 are included
#instead of just the 5030 most frequently occuring words. it ends up being the most
#frequently occuring 5000 words plus the other 30 words in table RG65 that were not
#already in the top 5030.
#comment the below code chunk out to keep the model using just the most frequent 5030 words
ctr = 0
for x,y in P:
	if x != 'serf':
		if x != uni[unigram[x]][0]:
			uni[5000+ctr] = (x, old[x])
			unigram[uni[5000+ctr][0]] = 5000+ctr
			ctr += 1
		if y != uni[unigram[y]][0]:
			uni[5000+ctr] = (y, old[y])
			unigram[uni[5000+ctr][0]] = 5000+ctr
			ctr += 1



#STEP 3 word-context vector M1 based on bigram counts; modified to count both preceding and following words
M1 = np.zeros(shape=(5030,5030))
for i in range(len(words) - 1):
    wi = re.sub('[^A-Za-z]+', '', words[i]).lower()
    wi1 = re.sub('[^A-Za-z]+', '', words[i+1]).lower()
    if wi != '' and wi1 != '' and wi == uni[unigram[wi]][0] and wi1 == uni[unigram[wi1]][0]:
        M1[unigram[wi], unigram[wi1]] += 1
        M1[unigram[wi], unigram[wi1]] += 1


#STEP 4 PPMI for M1 denoted M1plus
M1plus = np.zeros(shape=(5030,5030))
for i in range(5030):
	for j in range(5030):
		M1plus[i, j] = max(math.log((M1[i, j] / sum) / ((uni[i][1] / sum) * (uni[j][1] / sum) + 1e-31) + 1e-31, 2.0), 0)


#STEP 5 latent semantic model using SVD. M2_10, M2_50, and M2_100 denote
#truncated dimensions of 10, 50, 100 respectively
A, D, Q = np.linalg.svd(M1plus, full_matrices=False)
M2_10 = A[:, :10]
M2_50 = A[:, :50]
M2_100 = A[:, :100]

#STEP 6 done at beginning

#STEP 7 cosine similiarities for M1 (SM1), M1plus(SM1plus), M2_10 (SM2_10), M2_50 (SM2_50), M2_100 (SM2_100)
#a in front of name denotes matrix has cosine similarity for all pairs of words, later we pick relevant pairs
aSM1 = cosine_similarity(M1)
aSM1plus = cosine_similarity(M1plus)
aSM2_10 = cosine_similarity(M2_10)
aSM2_50 = cosine_similarity(M2_50)
aSM2_100 = cosine_similarity(M2_100)

#pick out the cosine similarity scores for the relevant pairs in P.
#SL only includes scores from S for pairs of words which actually exist in our top 5030 (so we have data)
#since I later forced all words in table RG65 into the top 5030, SL will contain all scores from S, except
#note the word 'serf' does not occur at all in the Brown Corpus, so its pair was omitted from analysis
L = []
SL = []
for i in range(len(P)):
	x,y = P[i]
	if x != 'serf' and x == uni[unigram[x]][0] and y == uni[unigram[y]][0]:
		L.append((x, y))
		SL.append(S[i])

SM1 = []
SM1plus = []
SM2_10 = []
SM2_50 = []
SM2_100 = []
for x,y in L:
	SM1.append(aSM1[unigram[x], unigram[y]])
	SM1plus.append(aSM1plus[unigram[x], unigram[y]])
	SM2_10.append(aSM2_10[unigram[x], unigram[y]])
	SM2_50.append(aSM2_50[unigram[x], unigram[y]])
	SM2_100.append(aSM2_100[unigram[x], unigram[y]])


#STEP 8 Pearson correlation. outputs tuple (Pearson coefficient, 2-tailed p value)
print("Cosine Similarities:")
print("S and SM1: ", pearsonr(SL, SM1))
print("S and SM1+: ", pearsonr(SL, SM1plus))
print("S and SM2_10: ", pearsonr(SL, SM2_10))
print("S and SM2_50: ", pearsonr(SL, SM2_50))
print("S and SM2_100: ", pearsonr(SL, SM2_100))


#Lab 1 extension Step 2 extract vectors for all pairs of words in Table 1 of RG65
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('word2vec_pretrain_vec/GoogleNews-vectors-negative300.bin', binary=True)
Mw = np.zeros(shape=(130,300))
for index, (i,j) in enumerate(P, start=0):
    Mw[index] = model[i]

for index, (i,j) in enumerate(P, start=0):
    Mw[index+65] = model[j]

#Step 3 calculate cosine similarities and report Pearson correlation with S
aSMw = cosine_similarity(Mw)
SMw = []
for i in range(len(P)):
    if P[i][0] != 'serf':
        SMw.append(aSMw[i][i+65])


print("S and SMw: ", pearsonr(SL, SMw))

#Step 4 Load analogy data from file
file = open('word-test.v1.txt', 'r')
text = file.read()
lines = text.split('\n')
words2 = []
for i in lines:
    if i != '' and i[0] != '/' and i[0] != ':':
        words2.append(i.split())


#Keep only the analogy tuples that have all relevant words in them (i.e. all 4 words in the
#analogy are part of our most common 5030 words from above), so we can use same set on LSA.
#Unfortunately, since the LSA model was built by converting everything to lowercase letters,
#anything with a capital letter such as city and country names will not be included.
rel_words = []
for w in words2:
    try:
        if w[0] == uni[unigram[w[0]]][0] and w[1] == uni[unigram[w[1]]][0] and w[2] == uni[unigram[w[2]]][0] and w[3] == uni[unigram[w[3]]][0]:
            rel_words.append(w)
    except KeyError:
        pass

#The below code will perform the analogy test for LSA on semantic analogy tuples. We had 90
#semantic relevant analogy tuples left from the original data and they are the first 90
#instances in rel_words. It counts how many times LSA pick the right word for the analogy.
#(Note: picks the word from the pool of 5030 whose vector is closest in cosine distance to the
#added vectors)
Mw3 = np.zeros(shape=(5031,100))

for index, (i, j) in enumerate(uni, start=0):
    try:
        Mw3[index] = M2_100[unigram[i]]
    except:
        pass


cnt = 0
for ww in rel_words[0:89]:
    Mw3[len(Mw3)-1] = M2_100[unigram[ww[0].lower()]] - M2_100[unigram[ww[1].lower()]] + M2_100[unigram[ww[3].lower()]]
    SMw3 = cosine_similarity(Mw3)
    max = -10
    maxind = -1
    for index, i in enumerate(SMw3[len(Mw3)-1], start=0):
        if i > max and index < 5030 and uni[index][0] != ww[3].lower():
            max = i
            maxind = index
    if ww[2] == uni[maxind][0]:
        cnt += 1

print('LSA semantic analogy accuracy: ', cnt, '/', 90)


#The below code will perform the analogy test for Word2Vec on the 90 semantic analogy tuples
Mw2 = np.zeros(shape=(5031,300))
for index, (i, j) in enumerate(uni, start=0):
    try:
        Mw2[index] = model[i]
    except:
        pass

cnt = 0
for ww in rel_words[0:89]:
    Mw2[len(Mw2)-1] = model[ww[0].lower()] - model[ww[1].lower()] + model[ww[3].lower()]
    SMw2 = cosine_similarity(Mw2)
    max = -10
    maxind = -1
    for index, i in enumerate(SMw2[len(Mw2)-1], start=0):
        if i > max and index < 5030 and uni[index][0] != ww[3].lower() and uni[index][0] != ww[0].lower() and uni[index][0] != ww[1].lower():
            max = i
            maxind = index
    if ww[2] == uni[maxind][0]:
        cnt +=1

print('Word2Vec semantic analogy accuracy', cnt, '/', 90)


#The below code will perform the analogy test for LSA on 90 of the syntactic analogy tuples
#We take every 10th element since we have many more than 90 syntactic tuples but want a good variety
#(and want to match the same number as taken in semantic)
Mw3 = np.zeros(shape=(5031,100))

for index, (i, j) in enumerate(uni, start=0):
    try:
        Mw3[index] = M2_100[unigram[i]]
    except:
        pass


cnt = 0
for ww in rel_words[90:980][0::10]:
    Mw3[len(Mw3)-1] = M2_100[unigram[ww[0].lower()]] - M2_100[unigram[ww[1].lower()]] + M2_100[unigram[ww[3].lower()]]
    SMw3 = cosine_similarity(Mw3)
    max = -10
    maxind = -1
    for index, i in enumerate(SMw3[len(Mw3)-1], start=0):
        if i > max and index < 5030 and uni[index][0] != ww[3].lower() and uni[index][0] != ww[0].lower() and uni[index][0] != ww[1].lower():
            max = i
            maxind = index
    if ww[2] == uni[maxind][0]:
        cnt += 1

print('LSA syntactic analogy accuracy: ', cnt, '/', 90)


#The below code will perform the analogy test for Word2Vec on 90 of the syntactic analogy tuples
Mw2 = np.zeros(shape=(5031,300))
for index, (i, j) in enumerate(uni, start=0):
    try:
        Mw2[index] = model[i]
    except:
        pass

cnt = 0
for ww in rel_words[90:980][0::10]:
    Mw2[len(Mw2)-1] = model[ww[0].lower()] - model[ww[1].lower()] + model[ww[3].lower()]
    SMw2 = cosine_similarity(Mw2)
    max = -10
    maxind = -1
    for index, i in enumerate(SMw2[len(Mw2)-1], start=0):
        if i > max and index < 5030 and uni[index][0] != ww[3].lower() and uni[index][0] != ww[0].lower() and uni[index][0] != ww[1].lower():
            max = i
            maxind = index
    if ww[2] == uni[maxind][0]:
        cnt +=1

print('Word2Vec syntactic analogy accuracy', cnt, '/', 90)
