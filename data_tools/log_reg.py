# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def log_loss(predictions,actual,eps=1e-15):
    '''take an array of prediction probabilities (clipped to avoid undefined values) and measures accuracy while
    also factoring for confidence'''
    #assert (max(predictions)<=1 and min(predictions)>=0), 'Please make sure to use predict_proba'
    
    preds_clipped = np.clip(predictions,eps,1-eps)
    
    loss = -1 * np.mean((actual * np.log(preds_clipped)) + ((1-actual) * np.log(1-preds_clipped)))
    
    return loss

def sigmoid(array):
    sig = 1 / (1 + np.exp(-array))
    return sig

class BinaryClassifier:
    
    def __init__(self,regularization=None):
        '''initializing the object with the option to select regularization'''
        
        '''Regularization will be a dict with type and lambda value'''
        
        if regularization is None:
            self.penalty_type = None
            self.penalty_lambda_ = 0
        
        else:
            self.penalty_type = list(regularization.keys())[0]
            self.penalty_lambda_ = regularization.get(self.penalty_type)
    
    def _gradient_descent(self,X,y,lr=.1,pandas=False,full_history=False,weights=None):
        
        if pandas or (isinstance(X,pd.DataFrame) & isinstance(y,pd.DataFrame)):
            self.X = X.values
            self.y = y.values
            self.Xnames = X.columns
            self.ynames = y.columns
        else:
            self.X = X
            self.y = y.reshape(len(y),1)
            self.Xnames = [i for i in range(X.shape[1])]
            
            
        '''learning rate for gradient descent algorithim'''    
        self.lr = lr
            
        m = len(self.X)
            
        n_features = self.X.shape[1]
        
        '''creating the weights, which will typically be all zeros'''
            
        if weights is None:
            self.weights = np.zeros(n_features)
        else:
            self.weights = weights
            
        if self.penalty_type is 'lasso':
            reg_loss = (self.penalty_lambda_)
            reg_gradient = (self.penalty_lambda_)
        
        elif self.penalty_type is 'ridge':
            reg_loss = ((self.penalty_lambda/2) * np.square(self.weights))/m
            reg_gradient = (self.penalty_lambda_/m)
            
        else:
            reg_loss = 0
            reg_gradient = 0
            
        weights_list = []
        scores_list = []
        
        for i in range(1000):
            
            weights = self.weights
            X = self.X
            lr = self.lr
            y = self.y
            
            '''p = prediction probabilities (0 < p < 1)'''
            
            p = sigmoid(np.dot(X, weights))
            
            error = p - y
            
            gradient = (np.dot(X.T,error) * lr) /m 
            
            weights = weights - gradient #/m + (reg_gradient)
            
            p = sigmoid(np.dot(X, weights))
            
            preds = np.round(p)
            
            loss = log_loss(p, y) + (reg_loss)
            
            auc = roc_auc_score(y, p)
             
            acc = accuracy_score(y,preds)
            
            weights_list.append(*weights)
            
            scores_list.append(auc,loss,acc)
            
            '''Early Stopping: if AUC does not change more than 0.01%, then break'''
            
            if i >50:
                if abs((scores_list[i][-3] - scores_list[i-50][-3]) / scores_list[i][-3]) < 0.0001:
                    break
       
        scores_df = pd.DataFame(scores_list,columns=['auc','loss','auc'])
        
        '''Finding the index with highest AUC score'''
        
        ID = scores_df.iloc[:,0].idxmax(axis=0)
        
        final_weights = weights_list[ID]
    #self.weights_final = weights_list[ID]
        
        weights_df = pd.DataFrame(weights_list,columns=self.Xnames)
        
        full_df = pd.concat([weights_df,scores_df],axis=1)
        
        return final_weights, full_df
            
        print('this test worked!')
            
        
        
        
        
        