#### Datawhale零基础入门NLP赛事 - Task4 基于fastText的文本分类

# Task4 基于fastText的文本分类


```python
import time
import numpy as np
import fasttext
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
```

## fastText分类


```python
def fasttext_model(nrows, train_num, lr=1.0, wordNgrams=2, minCount=1, epoch=25, loss='hs', dim=100):
    start_time = time.time()
    
    # 转换为FastText需要的格式
    train_df = pd.read_csv('C:/datalab/nlp/train_set.csv', sep='\t', nrows=nrows)

    # shuffle
    train_df = shuffle(train_df, random_state=666)

    train_df['label_ft'] = '__label__' + train_df['label'].astype('str')
    train_df[['text', 'label_ft']].iloc[:train_num].to_csv('C:/datalab/nlp/fastText_train.csv', index=None, header=None, sep='\t')

    model = fasttext.train_supervised('C:/datalab/nlp/fastText_train.csv', lr=lr, wordNgrams=wordNgrams, verbose=2, 
                                      minCount=minCount, epoch=epoch, loss=loss, dim=dim)

    train_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[:train_num]['text']]
    print('Train f1_score:', f1_score(train_df['label'].values[:train_num].astype(str), train_pred, average='macro'))
    val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[train_num:]['text']]
    print('Val f1_score:', f1_score(train_df['label'].values[train_num:].astype(str), val_pred, average='macro'))
    train_time = time.time()
    print('Train time: {:.2f}s'.format(train_time - start_time))

     # 预测并保存
    test_df = pd.read_csv('C:/datalab/nlp/test_a.csv')

    test_pred = [model.predict(x)[0][0].split('__')[-1] for x in test_df['text']]
    test_pred = pd.DataFrame(test_pred, columns=['label'])
    test_pred.to_csv('C:/datalab/nlp/test_fastText_ridgeclassifier.csv', index=False)
    print('Test predict saved.')
    end_time = time.time()
    print('Predict time:{:.2f}s'.format(end_time - train_time))
    
    
if __name__ == '__main__':  
    nrows = 200000
    train_num = int(nrows * 0.7)
    lr=0.01
    wordNgrams=2
    minCount=1
    epoch=25
    loss='hs'
    
    fasttext_model(nrows, train_num)
```

## K折交叉验证


```python
models = []
scores = []
pred_list = []
    
# K折交叉验证
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=666)
for train_index, test_index in skf.split(train_df['text'], train_df['label_ft']):
    
    train_df[['text', 'label_ft']].iloc[train_index].to_csv('C:/datalab/nlp/fastText_train.csv', index=None, header=None, sep='\t')

    model = fasttext.train_supervised('C:/datalab/nlp/astText_train.csv', lr=lr, wordNgrams=wordNgrams, verbose=2, 
                                          minCount=minCount, epoch=epoch, loss=loss)
    models.append(model)
        
    val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[test_index]['text']]
    score = f1_score(train_df['label'].values[test_index].astype(str), val_pred, average='macro')
    print('score', score)
    scores.append(score)
       
print('mean score: ', np.mean(scores))
train_time = time.time()
print('Train time: {:.2f}s'.format(train_time - start_time))
```
