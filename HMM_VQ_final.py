#Script to learn sequences of integers within a VQ vector that correspond to calls. 

from __future__ import print_function
import math
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from hmmlearn.hmm import GaussianHMM 
import seqlearn.hmm 

#Global variables
k = 2   	#number of classes
num_comp = 2    #number of HMM states (number_components parameter)

def main():

    #Read in our labels
    Z = []
    inp = open('byfile/Lnew.txt')
    for t in inp:
        for i in t.split(','):
            #Z.append(int(i) - 1)
            if (int(i) == 99):
                Z.append(1)
            else:
               Z.append(0)
    Z = np.array(Z)


    #Read in our data (observed sequences) 
    X_orig = []
    inp = open('byfile/Onew.txt')
    count = 0
    for t in inp:
        #if (int(Z[count]) == 98):
            #if (not(int(Z[count]) == 61 or int(Z[count] == 66))):
                #count += 1
                #continue
        #if (not (int(Z[count]) == 8)):
	#if ((int(Z[count]) == 67) or( int(Z[count]) == 62)): 
        X_orig.append([])
	for i in t.split():
	    X_orig[-1].append(float(i))
	    #if (int(Z[count]) == 61): 
	        #X_orig[-1].append(int(0))
            #if ( int(Z[count]) == 66): 
	        #X_orig[-1].append(int(1))
	X_orig[-1].append(int(Z[count]))
        count += 1

    X_orig = np.array(X_orig)


    #count the entire distribution
    #count = [0]*k
    #print(len(X_orig))
    #for O in X_orig:
        #index = int(O[len(O)-1])
        #count[index] += 1
    #print("distribution of entire data set ",count)


    #X, Y = cross_validation.train_test_split(X_orig, test_size=0.2) 

    #KFolds
    kf_num =  10 
    kf = KFold(len(X_orig), n_folds=kf_num)

    #Totals for final averages
    total_acc = 0
    total_prec =[0]*k
    total_rec = [0]*k

    for train, test in kf:
        X = X_orig[train]
        Y = X_orig[test]

	#Partition classes by label
	C = [[] for i in range(k)]
	Z_c = [[] for i in range(k)]

	for x in X:
	    index = int(x[len(x) - 1]) 
	    C[index].append(x)
	    Z_c[index].append(index)


	#Train a model for each class
	C = np.array(C)
	M = [None]*k
	model_count = 0
	print(C.shape)
	for c in C:
	    t = np.array(c)
	    if (len(t) == 0):
	        model_count += 1
	        continue
	    t = t[:,:-1]
	    #t = np.squeeze(np.asarray(t))
	    t_new = []
	    t_new = np.array(t_new)
	    lengths = []
	    lengths = np.array(lengths)
	    for o in t:
	        for i in o:
	    	    t_new = np.concatenate([t_new, [int(i)]])            
   	        lengths = np.concatenate([lengths, [int(len(o))]])

		
	    t_new = t_new.reshape(-1,1)
	    lengths = np.squeeze(np.asarray(lengths))
	    lengths = lengths.reshape(1,-1)
	    lengths = lengths[0]
	    new_lengths = [None]*len(lengths)
	    for i in range(0, len(lengths)):
	        new_lengths[i] = (int(lengths[i]))
	    model = GaussianHMM(n_components=num_comp).fit(t_new, new_lengths)
	    M[model_count] = model
	    model_count += 1

         
	#Create test data and labels 
	labels = []

	#Count the distribution you trained on
	count = [0]*k
	for O in X:
            index = int(O[len(O)-1])
            count[index] += 1
	      
	print("distribution of train examples is ",count)

	#Classify
	for x in Y:
	    temp = x
	    temp = temp[:-1]
	    temp = np.squeeze(np.asarray(temp))
	    temp = temp.reshape(-1,1)
	    max_prob = float('-inf')
	    max_prob_index = -1
	    for i in range(0, len(M)):
		if (M[i] == None):
		    continue
		prob = M[i].score(temp)
		if (prob > max_prob):
		    max_prob = prob
		    max_prob_index = i
	    labels.append(max_prob_index)



	#Count the distribution of classified vs truth
	count = [0]*k
	print("labels ",len(labels))
	print("truth ",len(Y))
	for x in Y:
	    index = int(x[len(x)-1])
	    count[index] += 1

	count_l = [0]*k
	for l in labels:
	    index = int(l)
	    count_l[index] += 1

	print("distribution of truth test examples is ",count)
	print("distribution of classified test examples is ",count_l)



	#Compare true labels and classified labels 
	correct = 0
	accuracies = [0]*k 	  #true label for i & classified for i 
	size = [0]*k 		  #true label for i 
	size_classified = [0]*k   #classified for i 

	for i in range(0, len(labels)):
	    #print("orig number", int(Y[i, len(Y[i]) - 1]) )
	    index = int(Y[i, len(Y[i]) - 1]) 
	    #print("index", index)
	    size[index] += 1
	    index_classified = int(labels[i])
	    size_classified[index_classified] += 1
	    #print("true label ", index)
	    #print("predicted label ",labels[i])
	    l = labels[i]  
	    if (float(index) == float(l)):   #true label for i & classified for i
		correct += 1
		accuracies[index] += 1



	#Print results
	#accuracy = #correct classified/ total data points to classify 
	#precision class_i = #true label for i & classified for i / #classified for i
	#recall class_i =#true label for i & classified for i/ #true label for i

        
	print("Total accuracy is: ",float(correct)/float(len(Y)))
	print("\n")
	for i in range(0, len(accuracies)):
	    if (not(size[i] == 0)):
	        print("Recall of class ", i, " is: ",float(accuracies[i])/float(size[i]))
	    else:
	        print("Recall of class ", i, " is: 0. No true instances of this class exist in data")
	    if (not(size_classified[i] == 0)):
	        print("Precision of class ", i, " is: ",float(accuracies[i])/float(size_classified[i]))
	    else:
	        print("Precision of class ", i, " is: 0. No classified instances of this class detected in data")
	 
	    print("Number of true instances of this class: ", size[i])
   	    print("Number of classified instance of this class: ", size_classified[i])
	    print("Number of true and classified instance of this class: ", accuracies[i])
	    print("\n")

            if (not size_classified[i] == 0):
                total_prec[i] += float(accuracies[i])/float(size_classified[i])
            if (not size[i] == 0):
                total_rec[i] += float(accuracies[i])/float(size[i])

	#Print confusion matrix

	last_index = len(Y[0]) - 1
	y_true = Y[:, last_index]
	y_pred = labels 
	conf = confusion_matrix(y_true, y_pred) 

	print("The confusion matrix\n")
	print(conf)
        total_acc += float(correct)/float(len(Y)) 

    print("Average accuracy is ", float(total_acc/kf_num))
    for i in range(0, k):
        print("Average precision for class ", i, "is ", float(total_prec[i]/kf_num))
        print("Average recall for class ", i, "is ", float(total_rec[i]/kf_num))

if __name__ == "__main__":
    main()



