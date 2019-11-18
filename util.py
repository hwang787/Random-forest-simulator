from scipy import stats
import numpy as np


import time


# This method computes entropy for information gain
def entropy(class_y):
    # Input:            
    #   class_y         : list of class labels (0's and 1's)
    
    # TODO: Compute the entropy for a list of classes
    #
    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92
    entropy = 0
    n_a=np.count_nonzero(class_y)
    
    n_b=class_y.size-n_a
    if(n_a==0 or n_b==0):
        return 0
    p_a=n_a/class_y.size
    p_b= 1 - p_a
    entropy = -p_a*np.log2(p_a)-p_b*np.log2(p_b)
    return entropy


def partition_classes(X, y, split_attribute, split_val):
    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute
    
    # TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
    # 
    # You will have to first check if the split attribute is numerical or categorical    
    # If the split attribute is numeric, split_val should be a numerical value
    # For example, your split_val could be the mean of the values of split_attribute
    # If the split attribute is categorical, split_val should be one of the categories.   
    #
    # You can perform the partition in the following way
    # Numeric Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is less than or equal to the split value, and the 
    #   second list has all the rows where the split attribute is greater than the split 
    #   value. Also create two lists(y_left and y_right) with the corresponding y labels.
    #
    # Categorical Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all 
    #   the rows where the split attribute is equal to the split value, and the second list
    #   has all the rows where the split attribute is not equal to the split value.
    #   Also create two lists(y_left and y_right) with the corresponding y labels.

    '''
    Example:
    
    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]
    
    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.
    
    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.
    
    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]
              
    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.
        
    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]
              
    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]
               
    ''' 
    
    '''
    X_a=np.array(X)
    Y_a=np.array(y)
    if(type(split_val) == int):
        resl = [key for key, val in enumerate(X_a[:,split_attribute].tolist()) 
                      if int(val) <= int(split_val)]
        resr = [key for key, val in enumerate(X_a[:,split_attribute].tolist()) 
                      if int(val) > int(split_val)]
        
    else:
        resl = [key for key, val in enumerate(X_a[:,split_attribute].tolist()) 
                      if val == split_val]
        resr = [key for key, val in enumerate(X_a[:,split_attribute].tolist()) 
                      if val != split_val]
        
    
    
    X_left = X_a[resl,:]
    X_right = X_a[resr,:]
    
    y_left = Y_a[resl]
    y_right = Y_a[resr]
    '''
    try:
        selected = X[:, split_attribute] <= split_val
        X_left = X[selected]
        X_right = X[selected == False]
        y_left =y[selected]
        y_right = y[selected == False]
    except:
        resl = [key for key, val in enumerate(X_a[:,split_attribute].tolist()) 
                      if val == split_val]
        resr = [key for key, val in enumerate(X_a[:,split_attribute].tolist()) 
                      if val != split_val]
   
    
    
    return (X_left, X_right, y_left, y_right)

    
def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value
    
    # TODO: Compute and return the information gain from partitioning the previous_y labels
    # into the current_y labels.
    # You will need to use the entropy function above to compute information gain
    # Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf
    
    """
    Example:
    
    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]
    
    info_gain = 0.45915
    """
    info_gain = entropy(previous_y)
    for i in range(0,len(current_y)):
        info_gain = info_gain -current_y[i].size*entropy(current_y[i])/previous_y.size

    #info_gain = entropy(previous_y) - len(current_y[0])*entropy(current_y[0])/len(previous_y)- len(current_y[1])*entropy(current_y[1])/len(previous_y)
    return info_gain
    
#print(information_gain([0,0,0,1,1,1],[[0,0,0], [1,1,1]]))   
def findBestSplit(X,y):
    info_gain =-1
    best_attribute = 0
    best_val = 0
    for attribute in range(X.shape[1]):
        try:
            value = np.mean(X[:,attribute])
            #for value in np.array(X)[:,attribute]:
            X_left, X_right, y_left, y_right = partition_classes(X, y, attribute, value)
            _info_gain = information_gain(y,[y_left,y_right])
            if info_gain <= _info_gain:
                info_gain = _info_gain
                best_attribute = attribute
                best_val = value  
        except:
            value = X[:,attribute][0]
            
    return best_attribute, best_val