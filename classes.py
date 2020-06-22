
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier as KNN
from imblearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, confusion_matrix
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class Data_transformation(object):

    def __init__(self,sampling,dim_red,n_components_pca=None):
        self.sampling=sampling
        self.dim_red=dim_red
        self.n_components_pca=n_components_pca
        
    def do_dim_reduction(self):  #dimensionality reduction
        if self.dim_red=='PCA':
            self.dim=PCA(n_components=self.n_components_pca)
        elif self.dim_red=='LDA':
            self.dim=LDA(n_components=2)
        elif self.dim_red=='PCA+LDA':
            self.dim=PCA(n_components=30)
        elif self.dim_red=='unchanged':
            self.dim=None
        else:
            raise NotImplementedError('incorrect/unknown dimensionality reduction method')
        return self.dim
            
    def do_sampling(self):       #sampling
        if self.sampling=='smote':
            self.sampl=SMOTE()
        elif self.sampling=='undersampling':
            self.sampl=RandomUnderSampler()
        elif self.sampling=='oversampling':
            self.sampl=RandomOverSampler()
        elif self.sampling=='unchanged':
            self.sampl=None
        else:
            raise NotImplementedError('incorrect/unknown sampling method')
        return self.sampl
    
    def lda_after_pca(self):
        if self.dim_red=='PCA+LDA':
            self.dim_LDA_after_PCA=LDA(n_components=2)
            print('lda_after_pca is used')
        else:
            self.dim_LDA_after_PCA=None
        return self.dim_LDA_after_PCA    
        


class Classification(Data_transformation):
    
    
    def __init__(self,sampling,dim_red,n_components_pca=None,*args):       #*args - pass classfiers names as strings. here 'knn','rf' for k-nearest neighbor and random forest respectively
        super().__init__(sampling,dim_red,n_components_pca)
        
        self.classifiers=[] # creating a list of classifiers that were passed by user
        for arg in args:
            self.classifiers.append(arg)
        
    def split(self):
        self.X, self.y=make_classification(n_samples=300, n_features=150, n_informative=100, n_redundant=5, n_repeated=45,n_classes=3,n_clusters_per_class=1,weights=(0.6,0.2,0.2),random_state=2)
        self.X_train, self.X_test, self.y_train, self.y_test=tts(self.X,self.y, random_state=2) # random_state is used for reproducability (controls the shuffling)
    
# imblearn allows to call only transfrom method of the intermediary steps (for example PCA()) when imblearn.pipeline.Pipeline object's method predict is called.
# therefore the fit_resample method is not applied on neither test_set nor validate_test during gridsearch.
# below is the snippet from source code
 
    """
    Xt = X
       for _, _, transform in self._iter(with_final=False):
           if hasattr(transform, "fit_resample"):
               pass
           else:
               Xt = transform.transform(Xt)
       return self.steps[-1][-1].predict(Xt, **predict_params)
    """
    def classification_process(self):
        self.df=pd.DataFrame()
        print('new dataframe is created')
        switcher={
            'knn': self.clf_knn,
            'rf':self.clf_rf}
        for arg in self.classifiers:
            print(arg)
            func=switcher.get(arg,lambda:"classifier not implemented/not specified")
            func()
        self.df.to_excel('results_'+self.sampling+'_'+self.dim_red+'_'+str(self.n_components_pca)+'_.xlsx')
        
    #def save(self):
        #self.df.to_excel('results_'+self.sampling+'_'+self.dim_red+'_'+str(self.n_components_pca)+'_.xlsx')
        #print('results are saved')   
     
        
    def clf_knn(self):
        self.classifier='KNN'
         
        self.clf=KNN()
        self.pipeline=make_pipeline(self.do_dim_reduction(),self.lda_after_pca(),self.do_sampling(), self.clf)
        if self.sampling=="smote": #in case of smote we are trying to tune the number of neighbors
            param_grid=dict(smote__m_neighbors=[2,3,4,5,7,9],kneighborsclassifier__n_neighbors=[2,3,5,10,20])
        else:
            param_grid=dict(kneighborsclassifier__n_neighbors=[2,3,5,10,20])
            
        self.grid=RandomizedSearchCV(estimator=self.pipeline, param_distributions=param_grid, cv=5, n_iter=5, scoring='balanced_accuracy', random_state=2)
        self.grid.fit(self.X_train,self.y_train)
        self.y_predicted=self.grid.predict(self.X_test)
        
        self.calculate_metrics()
        self.store_results()
        
    
    def clf_rf(self):
        self.classifier='Random Forest'
        
        self.clf=RandomForestClassifier(random_state=2)
        self.pipeline=make_pipeline(self.do_dim_reduction(),self.lda_after_pca(),self.do_sampling(), self.clf)
        if self.sampling=="smote": #in case of smote we are trying to tune the number of neighbors
            param_grid=dict(smote__m_neighbors=[2,3,4,5,7,9],randomforestclassifier__criterion=['gini','entropy'],randomforestclassifier__max_depth=[2,3,5,10,20],randomforestclassifier__max_features=[2,3])
        else:    
            param_grid=dict(randomforestclassifier__criterion=['gini','entropy'],randomforestclassifier__max_depth=[2,3,5,10,20],randomforestclassifier__max_features=[2,3])
        self.grid=RandomizedSearchCV(estimator=self.pipeline, param_distributions=param_grid, cv=5, n_iter=5, scoring='balanced_accuracy', random_state=2)
        self.grid.fit(self.X_train,self.y_train)
        self.y_predicted=self.grid.predict(self.X_test)
        
        #self.results=self.grid.cv_results_
        #self.best_estimator=self.grid.best_estimator_
        #self.best_score=self.grid.best_score_
        #self.best_params=self.grid.best_params_
        
        self.calculate_metrics()
        self.store_results()
        

    def pass_func(self):
        pass    
    
    def calculate_metrics(self):
        self.matthews_corr_coeff=matthews_corrcoef(self.y_test,self.y_predicted)
        self.accuracy=accuracy_score(self.y_test,self.y_predicted)
        self.confusion_matrix=confusion_matrix(self.y_test, self.y_predicted)
        return self
         
    def store_results(self):
        self.d = []      
        self.d.append({
        'classifier': self.classifier,
        'sampling': self.sampling,
        'dimensionality_reduction':self.dim_red,
        'n_components_of_pca':self.n_components_pca,
        'mattews_corr_coeff':self.matthews_corr_coeff,
        'accuracy':self.accuracy})
        if self.df.empty==True:
            self.df=pd.DataFrame(self.d)
        else:
            self.df=self.df.append(self.d)
        return self   
