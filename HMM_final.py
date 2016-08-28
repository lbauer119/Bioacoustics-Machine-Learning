from __future__ import print_function
import math
import csv
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from hmmlearn.hmm import GaussianHMM 
import seqlearn.hmm 


#Global variables
k = 3		#number of classes
num_comp = 2	#number of HMM states (number_components parameter)



def main():

    #Read in our labels
    Z = []
    inp = open('SW1_labels_call_type.txt')
    for t in inp:
        for i in t.split(','):
            Z.append(int(i))
    Z = np.array(Z)


    #Read in our lengths of each call
    L = []
    inp = open('SW1_labels_events.txt')
    for t in inp:
        for i in t.split(','):
            L.append(int(i))
    L = np.array(L)



    #Read in our data (observed sequences) 
    X_orig = []
    inp = open('SW1.txt')
    count_length = 0
    count_call = 0
    L_curr = 0
    L_orig = 0
    total_index = 0
    for t in inp:
        if (L_curr == 0):
            L_curr = L[count_length] 
            L_orig = L[count_length] 
            X_orig.append([])
            count_length += 1
        X_call = []
        for j in t.split():
	    X_call.append(float(j))
        X_call.append(int(L_orig))
        X_call.append(int(Z[count_call]))
        X_call.append(int(total_index))
        X_orig[-1].append(X_call)
        count_call += 1 
        L_curr -= 1
        L_orig = -1
        total_index += 1

    X_orig = np.array(X_orig)


    #KFolds
    kf_num =  5 
    kf = KFold(len(X_orig), n_folds=kf_num)
    kf_count = 0

    #Totals for final averages
    total_acc = 0
    total_prec =[0]*k
    total_rec = [0]*k
 

    #Prints to csv file for all iterations of kfolds
    with open('results.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
 

        for train, test in kf:
            X_o = (X_orig[train])
            Y_o = (X_orig[test])
            X = unshared_copy(X_o)
   	    Y = unshared_copy(Y_o)

            
            #Partition classes by label 
	    C = [[] for i in range(k)]
	    L_c = [[] for i in range(k)]

 	    for x in X:
	        index = int(x[0][len(x[0]) - 2]) 
                C[index].append(x)


	    #Train a model for each class
	    M = [None]*k
	    model_count = 0
	
  	    for c in C:
                t = unshared_copy(c) 
		if (len(t) == 0):
		    model_count += 1
		    continue
                for r in t:
		    for l in r:
			del l[len(l)-1]
			del l[len(l)-1]
			del l[len(l)-1]
       
         	t_new = []
		lengths = []
		lengths = np.array(lengths)
            
                for call in t:
	    	    for obs in call:
		        t_new.append(obs)            
		    lengths = np.concatenate([lengths, [int(len(call))]])
                        
		t_new = np.array(t_new)

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
	
            #Classify
            orig_labels = []
            feature_labels = []
            for x in Y:
                orig_labels.append(x[0][len(x[0]) - 2])
                feature_labels.append(x[0][len(x[0]) - 1])

            vector_index = [] 
            length_index = [] 
	    for x in Y:
                temp = []
                temp = unshared_copy(x)
                for r in temp:
	  	    del r[len(r) - 1]
                    vector_index.append(r[len(r)-1])
		    del r[len(r) - 1]
                    if (r[len(r) - 1] != -1):
                        length_index.append(r[len(r)-1])
		    del r[len(r) - 1]
	        #temp = np.squeeze(np.asarray(temp))
	        #t = t.reshape(1,-1)
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


            #Compare true labels and classified labels 
	    correct = 0
 	    accuracies = [0]*k 	        #true label for i & classified for i 
	    size = [0]*k 		#true label for i 
	    size_classified = [0]*k     #classified label for i 
        
  	    for i in range(0, len(labels)):
                index = int(orig_labels[i])
	        size[index] += 1
	        index_classified = int(labels[i])
	        size_classified[index_classified] += 1
	        l = labels[i]  
	        if (float(index) == float(l)):     #true label for i & classified for i
	            correct += 1
		    accuracies[index] += 1



  	    #Print results
	    #accuracy = #correct classified/ total data points to classify 
	    #precision class_i = #true label for i & classified for i / #classified for i
	    #recall class_i =#true label for i & classified for i/ #true label for i
        
            kf_count += 1
            print("RESULTS FOR KF ITERATION ", kf_count)
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
	    y_true = orig_labels 
	    y_pred = labels 
            print("Feature index begins at",feature_labels)
            print("Length of call is",length_index)
            print("Call numbers are ",test)
            print("True labels ", y_true)
            print("Predicted labels ",y_pred)
            print("\n")

	    conf = confusion_matrix(y_true, y_pred) 
	    print("The confusion matrix\n")
	    print(conf)
	    
            total_acc += float(correct)/float(len(Y)) 
            print("\n")
            print("--------------------------------------")

            for i in range(0, len(feature_labels)):
                writer.writerow([feature_labels[i], length_index[i], test[i], y_true[i], y_pred[i]])	
            writer.writerow("\n")
        

        print("Average accuracy is ", float(total_acc/kf_num))
        for i in range(0, k):
            print("Average precision for class ", i, "is ", float(total_prec[i]/kf_num))
            print("Average recall for class ", i, "is ", float(total_rec[i]/kf_num))



def unshared_copy(inList):
    if isinstance(inList, list):
	return list (map(unshared_copy, inList) )
    return inList



if __name__ == "__main__":
    main()

