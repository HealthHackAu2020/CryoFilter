# For GBDT
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot as plt

class GBDTWrapper():
    def __init__(self, x, y, x_labels=None, params=None, random_state=0, test_size=0.3):
        '''
        A convenience wrapper for training a GBDT
        x: NxM array of N samples of M dependent variables
        y: N array of labels for each of the N samples
        
        '''
        self._trained = False
        
        self.x = x
        self.x_labels = x_labels
        self.Y = y
        
        self.random_state = random_state
        self.test_size = test_size
        
        # Normalise x data
        self._scaler = StandardScaler()
        if self.x_labels is not None:
            columns = [x+' normed' for x in self.x_labels]
        else:
            columns = None
        self.X = pd.DataFrame(self._scaler.fit_transform(self.x),
                              columns = columns)
        
        # Make test/train split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=self.test_size, random_state=self.random_state)
        
        # Convert to LGB format
        self.lgb_dataset = lgb.Dataset(self.X_train, label=self.y_train)
        
    
    def train(self, epochs=10000, params=None, random_state=0):
        if params is None:
            params={
            "objective" : "binary",
            "metric" : "auc",
            "boosting": 'gbdt',
            "max_depth" : -1,
            "learning_rate" : 0.01,
            "verbosity" : 1,
            "seed": random_state
            }
        self.params = params
        
        print('Training GBDT . . .')
        self.model = lgb.train(self.params, self.lgb_dataset, epochs)
        self._trained = True
        print('Done!')
        
        
    def plotROC(self):
        if not self._trained:
            raise ValueError("Need to train model first")
        
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(self.y_test))]

        # predict probabilities
        lr_probs = self.model.predict(self.X_test)

        # calculate scores
        ns_auc = roc_auc_score(self.y_test, ns_probs)
        lr_auc = roc_auc_score(self.y_test, lr_probs)


        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_test, lr_probs)

        # plot the roc curve for the model
        plt.figure(figsize=(5,5), dpi= 200)
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # show the plot
        plt.title('Logistic: ROC AUC=%.3f' % (lr_auc), fontweight='bold')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.grid()
        plt.show()