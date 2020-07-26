from tqdm import tqdm
import numpy as np

# For featurizer
import itertools
from matplotlib import pyplot as plt
from sklearn import decomposition
from time import time
from sklearn.preprocessing import StandardScaler

class Featurizer():
    def __init__(self, data, n_components=9):
        '''
        Takes a length-N list (data) of equally-sized numpy arrays with M elements,        
        Calculates features on the flattened data (where each entry in the list is 
        interpreted as a sample of the M features)
        
        Example:
        
        > F = Featurizer(data, n_components = 8)
        > F.fit()
        
        F.features: length 3*n_components list of features
        F.feature_coeffs: [N, 3*n_components] array of feature coefficients for each sample
        F.feature_labels: length 3*n_components list of feature labels
        '''
        self._raw_data = data
        self.n_components = n_components
        
        self._preprocessed = False
        self._estimators_estimated = False
        self._features_featurized = False
        
    def fit(self):
        # Flatten and Normalise data into contiguous array
        print("Preprocessing data . . .")
        self.preprocessData()
        
        # Fit estimators to data
        print("Fitting estimators . . .")
        self.getEstimators()
        
        # Calculate features
        print("Calculating features . . .")
        self.getFeatures()
        
        print("Done!")
    
    def preprocessData(self):
        # Stack into contiguous array
        data = np.stack(self._raw_data, axis=0)
        
        # Flatten
        self._raw_data_shape = data.shape
        data = data.reshape(self._raw_data_shape[0], -1)
        
        # Zero-mean and unit-variance rescaling
        self._scaler = StandardScaler()
        self._scaler.fit(data)
        self.data = self._scaler.transform(data)
        
        self._preprocessed = True
        
    
    def getEstimators(self):
        '''
        Makes list of ('name', estimator) pairs for PCA, ICA, FA
        and fits estimatorsto data
        '''
        if not self._preprocessed:
            raise ValueError("Data must be preprocessed and estimators constructed")
            
        self._estimators = [
            ('PCA',
             decomposition.PCA(n_components=self.n_components, svd_solver='randomized',
                               whiten=True)),
            ('FastICA',
             decomposition.FastICA(n_components=self.n_components, whiten=True)),

            ('FactorAnalysis',
             decomposition.FactorAnalysis(n_components=self.n_components, max_iter=20))
        ]
            
        for name, estimator in self._estimators:
            print("Calculating %d features using %s..." % (self.n_components, name))
            t0 = time()
            estimator.fit(self.data)
            train_time = (time() - t0)
            print("\tTime taken = %0.3fs" % train_time)
            
            
        self._estimators_estimated = True
        
        
    def getFeatures(self):
        '''
        Calculates coefficients of data with respect to each estimator
        '''
        if not self._estimators_estimated:
            raise ValueError("Estimators must be fitted to data firts")
            
        #self._coeffs = {}
        features = []
        feature_coeffs = []
        feature_labels = []
        
        for name, estimator in self._estimators:
            features.append(estimator.components_.reshape(self.n_components,*self._raw_data_shape[1:]))
            
            coeffs = estimator.transform(self.data)
            #coeffs = np.matmul(estimator.components_, self.data.T).T
            feature_coeffs.append(coeffs)
            
            labels = []
            for i in range(self.n_components):
                labels.append("{} {}".format(name, i))
            feature_labels.append(labels)
        
        self.features = list(itertools.chain.from_iterable(features))
        self.feature_coeffs = np.concatenate(feature_coeffs, axis=1)
        self.feature_labels = list(itertools.chain.from_iterable(feature_labels))
        
        self._features_featurized = True
        
        
    def plot2DComponents(self, n_col = 3, cmap=plt.cm.gray):
        '''
        Makes a figure showing the components identified by each estimator
        Note that this will not work for non-image data
        
        n_col: number of columns in each plotted figure
        cmap: colormap to use for plotted 2D components
        '''
        
        # check that we're using 3D data
        if len(self._raw_data_shape)!=3:
            raise ValueError("Cannot plot 2D components for non-2D data")
            
        # Check that we've actually calculated features
        if not self._estimators_estimated:
            raise ValueError("Estimators need to be fitted to data before plotting")
        
        n_row = int(np.ceil(self.n_components/n_col))
        image_shape = (self._raw_data_shape[1], self._raw_data_shape[2])
        
        for name, estimator in self._estimators:
            plt.figure(figsize=(2. * n_col, 2.26 * n_row))
            plt.suptitle(name, size=16)
            for i, comp in enumerate(estimator.components_):
                plt.subplot(n_row, n_col, i + 1)
                vmax = max(comp.max(), -comp.min())
                plt.imshow(comp.reshape(image_shape), cmap=cmap,
                           interpolation='nearest',
                           vmin=-vmax, vmax=vmax)
                plt.xticks(())
                plt.yticks(())
            plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
            
            
class FeatureEnsembler():
    def __init__(self, data, transforms, n_components=9):
        '''
        Constructs an aggregated set of features of (data) by applying the transforms
        in (transforms)
        
        data: should be a length N list of numpy arrays of identical size
        transforms: should be a length P list of ImageTransform objects
        
        '''
        
        self.data = data
        self.transforms = transforms
        self.n_components = n_components
        
    def fit(self):
        self.getAllFeatures()
        
    
    def getAllFeatures(self):
        features = []
        feature_coeffs = []
        feature_labels = []
        for i, transform in enumerate(self.transforms):
            data_tf = transform.apply(self.data)
            
            F = Featurizer(data_tf, n_components=self.n_components)
            F.fit()
            features.append(F.features)
            feature_coeffs.append(F.feature_coeffs)
            feature_labels.append([transform.name+": "+x for x in F.feature_labels])
            
            
        self.features = list(itertools.chain.from_iterable(features))
        self.feature_coeffs = np.concatenate(feature_coeffs, axis=1)
        self.feature_labels = list(itertools.chain.from_iterable(feature_labels))