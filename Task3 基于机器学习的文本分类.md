#### Datawhale零基础入门NLP赛事 - Task3 基于机器学习的文本分类

# Task3 基于机器学习的文本分类


```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

```


```python
train = pd.read_csv('C:/datalab/nlp/train_set.csv', sep='\t')
test = pd.read_csv('C:/datalab/nlp/test_a.csv', sep='\t')
```


```python
train_text = train['text']
test_text = test['text']
y_train = train['label']
```


```python
#TF-IDF
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)

word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
train_word_features
```




    <200000x6977 sparse matrix of type '<class 'numpy.float64'>'
    	with 56074040 stored elements in Compressed Sparse Row format>




```python
##逻辑回归
X_train = train_word_features

# 可以改变输入维度
x_train_, x_valid_, y_train_, y_valid_ = train_test_split(X_train, y_train, test_size=0.3,random_state=420)
X_test = test_word_features

clf = LogisticRegression(C=3, solver='saga',n_jobs=-1)
clf.fit(x_train_, y_train_)

y_pred = clf.predict(x_valid_)
train_scores = clf.score(x_train_, y_train_)
print(train_scores, f1_score(y_pred, y_valid_, average='macro'))
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_sag.py:330: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      "the coef_ did not converge", ConvergenceWarning)
    

    0.9546785714285714 0.9188123406554928
    


```python
submission = pd.read_csv('C:/datalab/nlp/test_a_sample_submit.csv')
preds = clf.predict(X_test)
submission['label'] = preds
submission.to_csv('lr_submission.csv', index=False)
```


```python
#XGBOOST
import xgboost as xgb
train_matrix = xgb.DMatrix(x_train_ , label=y_train_, missing=np.nan)
valid_matrix = xgb.DMatrix(x_valid_ ,label=y_valid_ , missing=np.nan)
test_matrix  = xgb.DMatrix(X_test, missing=np.nan)
params = {'booster': 'gbtree',
          'objective':'multi:softmax',
          'min_child_weight': 5,
          'max_depth': 6,
          'subsample': 0.5,
          'num_class':14,
          'colsample_bytree': 0.5,
          'eta': 0.05,
          'seed': 2020,
          'nthread': 36,
          'silent': True,}
watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
model = xgb.train(params, train_matrix,num_boost_round=15000, evals=watchlist ,verbose_eval=500, early_stopping_rounds=1000)
val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)
test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)

```

    [0]	train-merror:0.589869	eval-merror:0.59465
    Multiple eval metrics have been passed: 'eval-merror' will be used for early stopping.
    
    Will train until eval-merror hasn't improved in 1000 rounds.
    [500]	train-merror:0.302919	eval-merror:0.341825
    [1000]	train-merror:0.2724	eval-merror:0.335925
    [1500]	train-merror:0.249525	eval-merror:0.33385
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-24-72eb03932e41> in <module>
         15           'silent': True,}
         16 watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
    ---> 17 model = xgb.train(params, train_matrix,num_boost_round=15000, evals=watchlist ,verbose_eval=500, early_stopping_rounds=1000)
         18 val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)
         19 test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)
    

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, learning_rates)
        214                            evals=evals,
        215                            obj=obj, feval=feval,
    --> 216                            xgb_model=xgb_model, callbacks=callbacks)
        217 
        218 
    

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\training.py in _train_internal(params, dtrain, num_boost_round, evals, obj, feval, xgb_model, callbacks)
         72         # Skip the first update if it is a recovery step.
         73         if version % 2 == 0:
    ---> 74             bst.update(dtrain, i, obj)
         75             bst.save_rabit_checkpoint()
         76             version += 1
    

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\core.py in update(self, dtrain, iteration, fobj)
       1107         if fobj is None:
       1108             _check_call(_LIB.XGBoosterUpdateOneIter(self.handle, ctypes.c_int(iteration),
    -> 1109                                                     dtrain.handle))
       1110         else:
       1111             pred = self.predict(dtrain)
    

    KeyboardInterrupt: 



```python
submission = pd.read_csv('C:/datalab/nlp/test_a_sample_submit.csv')
submission['label'] = test_pred
submission.to_csv('xgb_submission.csv', index=False)
```
