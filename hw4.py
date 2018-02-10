#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn import svm
from nltk.stem.porter import *
from sklearn import linear_model
import scipy.optimize
import nltk
import string
import math
import operator
from textblob import TextBlob as tb
from collections import defaultdict, Counter


def main():
	
	#read in data from pickle
	data = pd.read_pickle("./beer.pkl")
	reviews = data['review/text'][:5000]
	review_overall = data['review/overall']

#Question 1:
	print("Question 1:")
	punctuation = set(string.punctuation)
	bigram = defaultdict(int)
	prev = False
	prevWord = str()

	for review in reviews:
		r = ''.join([c for c in review.lower() if not c in punctuation])
		for w in r.split():
			w.replace('\t', '')
			if(prev):
				words = prevWord + ' ' + w    
				bigram[words]+=1        
				prevWord = w
			else:
				prevWord = w
				prev= True

	Bigrams = sorted(bigram.items(), key=operator.itemgetter(1))
	Bigrams.reverse()

	for word, count in Bigrams[:5]:
		print(word, "occurences:", count)
	
#Question 2:
	print("Question 2:")
	top1000bigrams = Bigrams[:1000]
	top_bigrams_ID = dict(zip(top1000bigrams, range(len(top1000bigrams))))
	X = [feature(review, top_bigrams_ID, top1000bigrams) for review in reviews]
	y = [d for d in review_overall]
	y = y[:5000]
	clf = linear_model.Ridge(1, fit_intercept=False)
	clf.fit(X,y)
	theta = clf.coef_
	predictions = clf.predict(X)
	mse = MSE(y,predictions)
	print("question 2 MSE: ", mse)

#Question 3:
	print("Question 3:")
	unigram = defaultdict(int)
	for review in reviews:
		r = ''.join([c for c in review.lower() if not c in punctuation])
		for w in r.split():
			w.replace('\t', '')
			unigram[w]+=1

	unigram_bigram = {**unigram, **bigram}

	uni_bigrams_sorted = sorted(unigram_bigram.items(), key=operator.itemgetter(1))
	uni_bigrams_sorted.reverse()

	top1000 = [x[0] for x in uni_bigrams_sorted[:1000]]

	wordId = dict(zip(top1000, range(len(top1000))))
	X1 = [featureUB(review, top1000, wordId) for review in reviews]
	
	clf1 = linear_model.Ridge(1, fit_intercept=False)
	clf1.fit(X1,y)
	theta1 = clf1.coef_
	predictions1 = clf1.predict(X1)
	mse1 = MSE(y,predictions1)
	print("question 3 MSE: ", mse1)

#Question 4:
	print("Question 4:")
	


	print("The top 5 positive features with their weights are: ")
	for ind in sorted(range(len(theta1)-1), key=lambda i: theta1[i])[-5:]:
		print(top1000[ind], theta1[ind])
	print("The lowest 5 features are: ")
	for ind in sorted(range(len(theta1)-1), key=lambda i: theta1[i])[:5]:
		print(top1000[ind], theta1[ind])


#Question 5:
	print("Question 5:")

	tbList = []
	for review in reviews:
	  	tbList.append(tb(review))


	foam_idf = idf('foam',tbList)	
	foam_tfidf = tfidf('foam', tbList[0], tbList)
	smell_idf = idf('smell', tbList)
	smell_tfidf = tfidf('smell', tbList[0], tbList)
	banana_idf = idf('banana', tbList)
	banana_tfidf = tfidf('banana',tbList[0], tbList)
	lactic_idf = idf('lactic', tbList)
	lactic_tfidf = tfidf('lactic', tbList[0], tbList)
	tart_idf = idf('tart',tbList)
	tart_tfidf = tfidf('tart',tbList[0], tbList)


	print("foam idf:", foam_idf, "tf-idf:",foam_tfidf)
	print("smell idf:",smell_idf, "tf-idf:", smell_tfidf)
	print("banana idf:", banana_idf, "tf-idf:", banana_tfidf)
	print("lactic idf:", lactic_idf,"tf-idf:", lactic_tfidf)
	print("tart idf:", tart_idf, "tf-idf:", tart_tfidf)

#Question 6:
	print("Question 6:")

	review1 =''.join([c for c in reviews[0].lower() if not c in punctuation])
	review2 =''.join([c for c in reviews[1].lower() if not c in punctuation])

	review1_words = set(review1.split())
	review2_words = set(review2.split())

	same_words = review1_words.intersection(review2_words)

	review1_tfidfs = []
	review2_tfidfs = []

	for word in same_words:
		word.replace('\t', '')
		review1_tfidfs.append(tfidf(word, tbList[0], tbList))
		review2_tfidfs.append(tfidf(word, tbList[1], tbList))

	cos_sim = 1 - scipy.spatial.distance.cosine(review1_tfidfs,review2_tfidfs)
	print("Cosine Similarity between first 2 reviews: ", cos_sim)

#Question 7:
	print("Question 7:")

	cos_sims = []
	ind = 1
	for review in reviews[1:]:
		r = ''.join([c for c in review.lower() if not c in punctuation])
		review_words = set(r.split())

		same = review1_words.intersection(review_words)

		review_tfidfs = []
		review1_tfidfs = []
		for word in same:
			word.replace('\t', '')
			review1_tfidfs.append(tfidf(word, tbList[0], tbList))
			review_tfidfs.append(tfidf(word,tbList[ind], tbList))
	
		cos_sim =  1 - scipy.spatial.distance.cosine(review1_tfidfs,review_tfidfs)
		ind+=1
		cos_sims.append((review, cos_sim))

	max_sim = max(cos_sims, key=operator.itemgetter(1))[1]
	max_review = max(cos_sims, key=operator.itemgetter(1))[0]
	print("Max Sim is:", max_sim, "and is between first review and:", max_review)


#Question 8:
	
	unigrams_sorted = sorted(unigram.items(), key=operator.itemgetter(1))
	uni_bigrams_sorted.reverse()
	X2 = []
	ind =0
	for review in reviews:
		feat = []
		for word, count in unigrams_sorted[:1000]:
			tf_idf = tfidf(word,tbList[ind], tbList)
			feat.append(tf_idf)
		ind+=1
		feat.append(1)
		X2.append(feat)


	clf2 = linear_model.Ridge(1, fit_intercept=False)
	clf2.fit(X2,y)
	theta2 = clf2.coef_
	predictions2 = clf1.predict(X2)
	mse2 = MSE(y,predictions2)
	print("question 8 MSE: ", mse2)
	

def featureUB(datum, top, ID):
	feat = [0]*1000
	punctuation = string.punctuation
	r = ''.join([c for c in datum.lower() if not c in punctuation])
	kWords = r.split()
	for index, w in enumerate(kWords):
		if index < len(kWords)-1:
			word = w + " " + kWords[index+1] ##bigrams
			if word in top:
				feat[ID[word]] += 1
			if w in top:
				feat[ID[w]] += 1
	feat.append(1) #offset
	return feat
def feature(datum, bigramID,top1000):
	feat = [0]*1000
	punctuation = string.punctuation
	

	r = ''.join([c for c in datum.lower() if not c in punctuation])
	kWords = r.split()
	for index, w in enumerate(kWords):
		if index < len(kWords)-1:
			word = w + " " + kWords[index+1] ##bigrams
		if word in top1000:
			feat[bigramID[word]] += 1
	feat.append(1)
	return feat
def MSE(data, predict):
	mse = 0
	for rating, prediction in zip(data, predict):
		mse +=(rating - prediction)**2

	return (mse/len(data))
def tf(word, b):
  	return float(float(b.words.count(word)) / float(len(b.words)))

def n_containing(word, blist):
  	return sum(1 for b in blist if word in b)

def idf(word, blist):
  	return float(math.log10(len(blist) / float((1 + n_containing(word, blist)))))

def tfidf(word, b, blist):
  	return float(tf(word, b) * idf(word, blist))

if __name__ == '__main__':
	main()