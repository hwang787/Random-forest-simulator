from util import entropy, information_gain, partition_classes, findBestSplit
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.isTree = True
        self.tree = {}
        self.info = []
        

    def learn(self, X, y):
        if X.min() == X.max():
            self.info=np.round(y.mean())
            self.isTree = False
        elif (entropy(y)==0):
            #print(y)
            self.info=y[0]
            #print(self.info)
            self.isTree = False
            #print("this is a leaf")
        
        else:
            
            best_attribute, best_val = findBestSplit(X,y)
            
            self.info = [best_attribute, best_val]
            
            X_left, X_right, y_left, y_right = partition_classes(X, y, best_attribute, best_val)

            self.tree["left"] = DecisionTree()
            self.tree["left"].learn(X_left, y_left)
            self.tree["right"] = DecisionTree()
            self.tree["right"].learn(X_right, y_right)
            
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        


    def classify(self, record):
        
        # TODO: classify the record using self.tree and return the predicted label
        if (self.isTree==False):
            #print(self.info)
            return self.info
        
        else:
            #y=[0.0]*len(record)
            
                
            #X_left, X_right, y_left, y_right = partition_classes(record, y, self.info[0], self.info[1])
            #print(record)
            #print(X_left)
            
            #print(self.info)
                
            if (record[self.info[0]]<=self.info[1]):
                
                return self.tree["left"].classify(record)
            else:
                return self.tree["right"].classify(record)
                    
                    
            #list1= self.tree["left"].classify(X_left)
            #list2= self.tree["right"].classify(X_right)
            #return list2+list1
        
