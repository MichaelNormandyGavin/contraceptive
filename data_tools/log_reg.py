# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def log_loss(predictions,actual,eps=1e-15):
    '''take an array of prediction probabilities (clipped to avoid undefined values) and measures accuracy while
    also factoring for confidence'''
    #assert (max(predictions)<=1 and min(predictions)>=0), 'Please make sure to use predict_proba'
    
    p_clipped = np.clip(predictions,eps,1-eps)
    
    loss = -1 * np.mean((actual * np.log(p_clipped)) + ((1-actual) * np.log(1-p_clipped)))
    
    return loss

def sigmoid(array):
    sig = 1 / (1 + np.exp(-array))
    return sig

class BinaryClassifier:
    
    def __init__(self,regularization=None):
        '''initializing the object with the option to select regularization
        
        Regularization will be a dict with type (ridge/lasso) and lambda value'''
        
        if regularization is None:
            self.penalty_type = None
            self.penalty_lambda_ = 0
        
        else:
            self.penalty_type = list(regularization.keys())[0]
            self.penalty_lambda_ = regularization.get(self.penalty_type)
    
    def _gradient_descent(self, X, y, lr=.1, pandas=False, full_history=False, weights=None, early_stopping=True):
        
        if pandas or (isinstance(X,pd.DataFrame) & isinstance(y,pd.DataFrame)):
            X = X.values
            y = y.values
            Xnames = X.columns
            ynames = y.columns
        else:
            X = X
            y = y
            Xnames = [i for i in range(X.shape[1])]
            
            
        '''learning rate for gradient descent algorithim'''    
        self.lr = lr
            
        m = len(X)
            
        n_features = X.shape[1]
        
        '''creating the weights, which will typically be all zeros'''
            
        if weights is None:
            self.init_weights = np.zeros(n_features)
        else:
            self.init_weights = weights
            
        if self.penalty_type is 'lasso':
            reg_loss = (self.penalty_lambda_/m)
            reg_gradient = (-2*self.penalty_lambda_/m)
        
        elif self.penalty_type is 'ridge':
            reg_loss = (self.penalty_lambda_/2)
            reg_gradient = (-2*self.penalty_lambda_/m)
            
        else:
            reg_loss = 0
            reg_gradient = 0
            
        weights_list = []
        scores_list = []
        
        weights = self.init_weights
        
        for i in range(1000):
            
            if self.penalty_type is 'ridge':
                gradient_suffix = reg_gradient * weights
                loss_suffix = np.sum(reg_loss * np.square(weights)/m)
                
            elif self.penalty_type is 'lasso':
                gradient_suffix = reg_gradient * np.where(weights==0,0,np.where(weights>0,1,-1))
                loss_suffix = np.sum(reg_loss * np.abs(weights)/m)
            
            else:
                gradient_suffix = 0
                loss_suffix = 0
            
            lr = self.lr
            
            '''p = prediction probabilities (0 < p < 1)'''
            
            p = sigmoid(np.dot(X, weights))
            
            error = p - y
            
            gradient = (np.dot(X.T,error) * lr) /m 
            
            weights = weights - gradient + gradient_suffix
            
            p = sigmoid(np.dot(X, weights))
            
            preds = np.round(p)
            
            loss = log_loss(p, y) + loss_suffix
            
            auc = roc_auc_score(y, p)
             
            acc = accuracy_score(y,preds)
            
            weights_list.append([*weights])
            
            scores_list.append([auc,loss,acc])
            
            '''Early Stopping: if AUC does not change more than 0.01%, then break'''
            if early_stopping:
                if i >50:
                    if abs((scores_list[i][-3] - scores_list[i-50][-3]) / scores_list[i][-3]) < 0.0001:
                        break
       
        scores_df = pd.DataFrame(scores_list,columns=['auc','loss','acc'])
        
        '''Finding the index with highest AUC score'''
        
        highest_auc = scores_df.iloc[:,0].idxmax(axis=0)
        
        final_weights = weights_list[highest_auc]
        
        #self.weights_final = weights_list[highest_auc]
        
        weights_df = pd.DataFrame(weights_list,columns=Xnames)
        
        full_df = pd.concat([weights_df,scores_df],axis=1)
        
        #self.final_weights, self.full_history = final_weights, full_df
        
        #print('this test worked!')
        
        return final_weights, full_df
    
    def fit(self, X, y, **kwargs):
        
        '''Function to apply gradient descent. Compatible with multi-output
        
        X: features array (pandas or numpy)
        y: labels array (pandas or numpy)
        **kwargs: args to feed into gradient descent function'''
        
        if (isinstance(X,pd.DataFrame) & isinstance(y,pd.DataFrame)):
            self.X = X.values
            self.y = y.values
            self.Xnames = X.columns
            self.ynames = y.columns
        else:
            self.X = X
            self.y = y
            self.Xnames = [i for i in range(X.shape[1])]
            self.ynames = [i for i in range(y.shape[1])]
            
        weights_all = []
        
        history = []
            
        for i in range(len(self.ynames)):
            
            print('fitting to target {}...'.format(self.ynames[i]))
            
            w, d = self._gradient_descent(X=X,y=self.y[:,i],**kwargs)
            
            weights_all.append(w)
            
            history.append(d)
            
        print('Complete!')
        
        self.fitted_weights = np.array(weights_all)
        self.history = history
        
    def predict_proba(self, X, return_pandas=True):
        '''Use the fitted weights to predict sigmoid probabilities for each target
        
        X: features array (pandas or numpy)'''
        
        assert self.fitted_weights is not None, "Please use fit method before attempting to predict"
        
        p = sigmoid(np.dot(X, self.fitted_weights.T))
        
        if return_pandas:
            p =pd.DataFrame(data=p,columns=self.ynames)
        
        self.probabilities_matrix = p
        
        return p      
            
    def predict(self, X, **kwargs):
        '''Use the fitted weights and probabilities to predict classification for each input
        
        X: features array (pandas or numpy)'''
        
        p = self.predict_proba(X=X,**kwargs)
        
        p = pd.DataFrame(p) if not isinstance(p,pd.DataFrame) else p
        
        max_p = p.idxmax(axis=1)
        
        wide_p = pd.get_dummies(max_p)
        
        wide_p.columns = self.ynames
        
        self.predictions_matrix = wide_p
            
            
            
        
            
        
        
        
        
        