# CS114 Spring 2020 Programming Assignment 4
# Part-of-speech Tagging with Hidden Markov Models

import os
import numpy as np
from collections import defaultdict
import math

class POSTagger():

    def __init__(self):
        self.pos_dict = {}
        self.word_dict = {}
        self.initial = None
        self.transition = None
        self.emission = None
        self.UNK = '<UNK>'
        self.k = .6

    '''
    Trains a supervised hidden Markov model on a training set.
    self.initial[POS] = log(P(the initial tag is POS))
    self.transition[POS1][POS2] =
    log(P(the current tag is POS2|the previous tag is POS1))
    self.emission[POS][word] =
    log(P(the current word is word|the current tag is POS))
    '''
    def train(self, train_set):
        # iterate over training documents
        words = []
        tags = []
        initials = []
        sentences = []
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    f= f.readlines()
                    for line in f:
                        if len(line)>3:

                            line = line.split()
                            #print(line)

                            for i in range(len(line)):
                                line[i] = line[i].split('/')
                                while len(line[i])>2:
                                    line[i][0] = line[i][0]+'/'+line[i][1]
                                    del line[i][1]
                            initials.append(line[0])
                            for item in line:
                                words.append(item[0].lower())
                                tags.append(item[1].lower())
                            sentences.append(line)
            words = sorted(list(set(words)))
        uniquetags = sorted(list(set(tags)))
        index = 0
        self.word_dict[self.UNK.lower()]=-1

        for word in words:
            self.word_dict[word] = index
            index += 1
        index = 0
        for tag in uniquetags :
            self.pos_dict[tag] = index
            index += 1
        self.initial = np.zeros((1, len(uniquetags)))
        self.initial = np.add(self.initial, self.k)
        #print(initials)
        for item in initials:
            #print(item)
            self.initial[0, self.pos_dict[item[1]]] += 1
        #print(initials)
        #print(self.word_dict)
        #print(self.pos_dict)


        self.emission = np.zeros((len(uniquetags), len(words)+1))

        self.emission = np.add(self.emission, self.k)
        self.emission[:, -1] += 1

        self.transition = np.zeros((len(uniquetags), len(uniquetags)))
        self.transition = np.add(self.transition, self.k)

        for sent in sentences:
            for i in range(1, len(sent)):
                self.transition[self.pos_dict[sent[i-1][1]],self.pos_dict[sent[i][1]]] += 1
            for word in sent:
                self.emission[self.pos_dict[word[1]], self.word_dict[word[0].lower()]] += 1
        for w in range(len(self.transition)):
            div = 1/np.sum(self.transition[w])
            self.transition[w] = np.multiply(self.transition[w], div)

        for w in range(len(self.emission)):
            div = 1/np.sum(self.emission[w])
            self.emission[w] = np.multiply(self.emission[w], div)
        self.initial = np.multiply(self.initial, 1/np.sum(self.initial))
        #print(self.emission)
        self.initial = np.log(self.initial)
        self.transition = np.log(self.transition)
        self.emission = np.log(self.emission)

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):


        #print(self.word_dict, sentence)
        for word in range(len(sentence)):
            if sentence[word] not in self.word_dict:
                sentence[word] = -1
            else:
                sentence[word] = self.word_dict[sentence[word]]
        #print(sentence)
        # initialization step
        v = np.zeros((len(self.pos_dict), len(sentence)))
        backpointer = np.zeros((len(self.pos_dict), len(sentence)))

        v[:,0]= self.initial[0,:]+self.emission[:,sentence[0]]
        for time in range(1, len(sentence)):
            for state in range(len(v)):

                x = v[:, time-1]+self.transition[:, state]+self.emission[state, sentence[time]]
                #print("MEOW", x)
                v[state,time] = max(x)
                y = np.where(x == np.max(x))[0]

                backpointer[state, time] = y[0]

        #print(v)
        #print(backpointer)

        best_path = []
        bestpathprob = (np.max(v[:,-1]))
        x=v[:,-1]
        x=np.where(x == np.max(x))[0]
        best_path.append(int(x))
        for time in range(len(sentence)-1, 0, -1):
            x = int(backpointer[x ,time])
            best_path.append(x)


        #print(best_path)
        best_path.reverse()
        for i in range(len(best_path)):
            for k, v in self.pos_dict.items():
                if best_path[i] == v:
                    best_path[i] = k
        print(best_path)
        return best_path

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of POS tags such that:
    results[sentence_id]['correct'] = correct sequence of POS tags
    results[sentence_id]['predicted'] = predicted sequence of POS tags
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                print(name)
                with open(os.path.join(root, name)) as f:
                    f= f.readlines()
                    index = 0
                    for line in f:
                        if len(line)>3:
                            results[name+"_"+str(index)]['correct']= []
                            line = line.split()

                            sentence=[]
                            for i in range(len(line)):
                                line[i] = line[i].split('/')

                                while len(line[i])>2:
                                    line[i][0] = line[i][0]+'/'+line[i][1]
                                    del line[i][1]
                                sentence.append(line[i][0].lower())
                                results[name+"_"+str(index)]['correct'].append(line[i][1])
                            results[name+"_"+str(index)]['predicted'] = self.viterbi(sentence)
                            index+=1
                            

        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        totalcorr = 0
        total = 0
        for item in results:
            corr = results[item]['correct']
            pred = results[item]['predicted']
            for i in range(len(pred)):
                total +=1
                if pred[i]==corr[i]:
                    totalcorr+=1
        accuracy = totalcorr/total
        return accuracy

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('brown/train')
    results = pos.test('brown/dev')
    #pos.train('data_small/train_small')
    #results = pos.test('data_small/test_small')
    print('Accuracy:', pos.evaluate(results))
