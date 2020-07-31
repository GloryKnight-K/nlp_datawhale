#### Datawhale零基础入门NLP赛事 - Task05：基于深度学习的文本分类2

#### https://github.com/datawhalechina/team-learning-nlp/blob/master/NewsTextClassification/Task5%20%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB2.md
####  https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.18.6406111aIKCSLV&postId=118268

# 基于深度学习的文本分类

#### 1. Word2Vec
####  2. TextCNN，TextRNN
####  3. HAN


## Word2Vec

####  word2vec的主要思路：通过单词和上下文彼此预测，对应的两个算法分别为：
####  Skip-grams (SG)：预测上下文
####  Continuous Bag of Words (CBOW)：预测目标单词

####  另外提出两种更加高效的训练方法：
####  Hierarchical softmax
####  Negative sampling

####  word2vec分为2部分：1. 建立模型 2. 通过模型获取嵌入词向量
####  input_word, skip_window => input得到outpu的概率分布
#### input, output的one-hot编码

## Skip-grams的引入：
#### 目的：减小权重矩阵规模
####  方法：
####  1. 将常见的单词组合（word pairs）或者词组作为单个“words”来处理
#### 2. 对高频次单词进行抽样来减少训练样本的个数
#### 3. 对优化目标采用“negative sampling”方法，这样每个训练样本的训练只会更新一小部分的模型权重，从而降低计算负担

## Hierarchical Softmax：结合霍夫曼树做softmax，可以代替从隐藏层到输出softmax层的映射，减少softmax概率的计算量。


## TextCNN, TextRNN

####  TextCNN利用CNN（卷积神经网络）进行文本特征抽取，不同大小的卷积核分别抽取n-gram特征，
####  卷积计算出的特征图经过MaxPooling保留最大的特征值，然后将拼接成一个向量作为文本的表示。

####  TextRNN利用RNN（循环神经网络）进行文本特征抽取，由于文本本身是一种序列，而LSTM天然适合建模序列数据。
####  TextRNN将句子中每个词的词向量依次输入到双向双层LSTM，分别将两个方向最后一个有效位置的隐藏层拼接成一个向量作为文本的表示。


## HAN 
####  - Hierarchical Attention Network for Document Classification
####  基于层级注意力，在单词和句子级别分别编码并基于注意力获得文档的表示，然后经过Softmax进行分类。
####  其中word encoder的作用是获得句子的表示，可以替换为上节提到的TextCNN和TextRNN，也可以替换为下节中的BERT。


```python
import logging
import random

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set seed 
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
```




    <torch._C.Generator at 0x10991ac30>




```python
# split data to 10 fold
fold_num = 10
data_file = './data/train_set.csv'
import pandas as pd


def all_data2fold(fold_num, num=10000):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]

    total = len(labels)

    index = list(range(total))
    np.random.shuffle(index)

    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)

        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))

    return fold_data


fold_data = all_data2fold(10)
```

    2020-07-28 17:06:09,732 INFO: Fold lens [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    


```python
# build train data for word2vec
fold_id = 9

train_texts = []
for i in range(0, fold_id):
    data = fold_data[i]
    train_texts.extend(data['text'])
    
logging.info('Total %d docs.' % len(train_texts))
```

    2020-07-28 17:06:13,755 INFO: Total 9000 docs.
    


```python
logging.info('Start training...')
from gensim.models.word2vec import Word2Vec

num_features = 100     # Word vector dimensionality
num_workers = 8       # Number of threads to run in parallel

train_texts = list(map(lambda x: list(x.split()), train_texts))
model = Word2Vec(train_texts, workers=num_workers, size=num_features)
model.init_sims(replace=True)

# save model
model.save("./word2vec.bin")
```

    2020-07-28 17:09:18,564 INFO: Start training...
    2020-07-28 17:09:20,430 INFO: collecting all words and their counts
    2020-07-28 17:09:20,439 INFO: PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    2020-07-28 17:09:21,475 INFO: collected 5295 word types from a corpus of 8191447 raw words and 9000 sentences
    2020-07-28 17:09:21,476 INFO: Loading a fresh vocabulary
    2020-07-28 17:09:21,532 INFO: effective_min_count=5 retains 4335 unique words (81% of original 5295, drops 960)
    2020-07-28 17:09:21,533 INFO: effective_min_count=5 leaves 8189498 word corpus (99% of original 8191447, drops 1949)
    2020-07-28 17:09:21,553 INFO: deleting the raw counts dictionary of 5295 items
    2020-07-28 17:09:21,554 INFO: sample=0.001 downsamples 61 most-common words
    2020-07-28 17:09:21,554 INFO: downsampling leaves estimated 7070438 word corpus (86.3% of prior 8189498)
    2020-07-28 17:09:21,569 INFO: estimated required memory for 4335 words and 100 dimensions: 5635500 bytes
    2020-07-28 17:09:21,569 INFO: resetting layer weights
    2020-07-28 17:09:22,416 INFO: training model with 8 workers on 4335 vocabulary and 100 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
    2020-07-28 17:09:23,423 INFO: EPOCH 1 - PROGRESS: at 25.19% examples, 1764562 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:24,425 INFO: EPOCH 1 - PROGRESS: at 48.97% examples, 1711600 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:25,437 INFO: EPOCH 1 - PROGRESS: at 73.74% examples, 1722804 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:26,457 INFO: EPOCH 1 - PROGRESS: at 93.06% examples, 1621144 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:26,766 INFO: worker thread finished; awaiting finish of 7 more threads
    2020-07-28 17:09:26,776 INFO: worker thread finished; awaiting finish of 6 more threads
    2020-07-28 17:09:26,778 INFO: worker thread finished; awaiting finish of 5 more threads
    2020-07-28 17:09:26,779 INFO: worker thread finished; awaiting finish of 4 more threads
    2020-07-28 17:09:26,787 INFO: worker thread finished; awaiting finish of 3 more threads
    2020-07-28 17:09:26,791 INFO: worker thread finished; awaiting finish of 2 more threads
    2020-07-28 17:09:26,792 INFO: worker thread finished; awaiting finish of 1 more threads
    2020-07-28 17:09:26,796 INFO: worker thread finished; awaiting finish of 0 more threads
    2020-07-28 17:09:26,797 INFO: EPOCH - 1 : training on 8191447 raw words (7022049 effective words) took 4.4s, 1604175 effective words/s
    2020-07-28 17:09:27,807 INFO: EPOCH 2 - PROGRESS: at 19.87% examples, 1369972 words/s, in_qsize 13, out_qsize 2
    2020-07-28 17:09:28,830 INFO: EPOCH 2 - PROGRESS: at 39.74% examples, 1378706 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:29,831 INFO: EPOCH 2 - PROGRESS: at 60.03% examples, 1390399 words/s, in_qsize 14, out_qsize 1
    2020-07-28 17:09:30,844 INFO: EPOCH 2 - PROGRESS: at 82.80% examples, 1438377 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:31,536 INFO: worker thread finished; awaiting finish of 7 more threads
    2020-07-28 17:09:31,548 INFO: worker thread finished; awaiting finish of 6 more threads
    2020-07-28 17:09:31,550 INFO: worker thread finished; awaiting finish of 5 more threads
    2020-07-28 17:09:31,562 INFO: worker thread finished; awaiting finish of 4 more threads
    2020-07-28 17:09:31,571 INFO: worker thread finished; awaiting finish of 3 more threads
    2020-07-28 17:09:31,574 INFO: worker thread finished; awaiting finish of 2 more threads
    2020-07-28 17:09:31,578 INFO: worker thread finished; awaiting finish of 1 more threads
    2020-07-28 17:09:31,580 INFO: worker thread finished; awaiting finish of 0 more threads
    2020-07-28 17:09:31,580 INFO: EPOCH - 2 : training on 8191447 raw words (7022205 effective words) took 4.8s, 1468853 effective words/s
    2020-07-28 17:09:32,585 INFO: EPOCH 3 - PROGRESS: at 22.37% examples, 1564461 words/s, in_qsize 13, out_qsize 2
    2020-07-28 17:09:33,591 INFO: EPOCH 3 - PROGRESS: at 45.73% examples, 1605243 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:34,593 INFO: EPOCH 3 - PROGRESS: at 68.42% examples, 1603143 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:35,610 INFO: EPOCH 3 - PROGRESS: at 88.62% examples, 1553437 words/s, in_qsize 14, out_qsize 1
    2020-07-28 17:09:36,180 INFO: worker thread finished; awaiting finish of 7 more threads
    2020-07-28 17:09:36,181 INFO: worker thread finished; awaiting finish of 6 more threads
    2020-07-28 17:09:36,189 INFO: worker thread finished; awaiting finish of 5 more threads
    2020-07-28 17:09:36,190 INFO: worker thread finished; awaiting finish of 4 more threads
    2020-07-28 17:09:36,202 INFO: worker thread finished; awaiting finish of 3 more threads
    2020-07-28 17:09:36,203 INFO: worker thread finished; awaiting finish of 2 more threads
    2020-07-28 17:09:36,208 INFO: worker thread finished; awaiting finish of 1 more threads
    2020-07-28 17:09:36,218 INFO: worker thread finished; awaiting finish of 0 more threads
    2020-07-28 17:09:36,219 INFO: EPOCH - 3 : training on 8191447 raw words (7022165 effective words) took 4.6s, 1515257 effective words/s
    2020-07-28 17:09:37,242 INFO: EPOCH 4 - PROGRESS: at 22.08% examples, 1514817 words/s, in_qsize 14, out_qsize 1
    2020-07-28 17:09:38,244 INFO: EPOCH 4 - PROGRESS: at 45.81% examples, 1602826 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:39,245 INFO: EPOCH 4 - PROGRESS: at 69.73% examples, 1632965 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:40,249 INFO: EPOCH 4 - PROGRESS: at 94.49% examples, 1654082 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:40,451 INFO: worker thread finished; awaiting finish of 7 more threads
    2020-07-28 17:09:40,452 INFO: worker thread finished; awaiting finish of 6 more threads
    2020-07-28 17:09:40,454 INFO: worker thread finished; awaiting finish of 5 more threads
    2020-07-28 17:09:40,454 INFO: worker thread finished; awaiting finish of 4 more threads
    2020-07-28 17:09:40,458 INFO: worker thread finished; awaiting finish of 3 more threads
    2020-07-28 17:09:40,470 INFO: worker thread finished; awaiting finish of 2 more threads
    2020-07-28 17:09:40,476 INFO: worker thread finished; awaiting finish of 1 more threads
    2020-07-28 17:09:40,483 INFO: worker thread finished; awaiting finish of 0 more threads
    2020-07-28 17:09:40,484 INFO: EPOCH - 4 : training on 8191447 raw words (7022374 effective words) took 4.3s, 1650453 effective words/s
    2020-07-28 17:09:41,490 INFO: EPOCH 5 - PROGRESS: at 22.60% examples, 1578729 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:42,491 INFO: EPOCH 5 - PROGRESS: at 46.24% examples, 1624105 words/s, in_qsize 14, out_qsize 1
    2020-07-28 17:09:43,493 INFO: EPOCH 5 - PROGRESS: at 69.32% examples, 1623678 words/s, in_qsize 15, out_qsize 0
    2020-07-28 17:09:44,495 INFO: EPOCH 5 - PROGRESS: at 92.82% examples, 1629617 words/s, in_qsize 14, out_qsize 1
    2020-07-28 17:09:44,782 INFO: worker thread finished; awaiting finish of 7 more threads
    2020-07-28 17:09:44,787 INFO: worker thread finished; awaiting finish of 6 more threads
    2020-07-28 17:09:44,794 INFO: worker thread finished; awaiting finish of 5 more threads
    2020-07-28 17:09:44,803 INFO: worker thread finished; awaiting finish of 4 more threads
    2020-07-28 17:09:44,819 INFO: worker thread finished; awaiting finish of 3 more threads
    2020-07-28 17:09:44,828 INFO: worker thread finished; awaiting finish of 2 more threads
    2020-07-28 17:09:44,832 INFO: worker thread finished; awaiting finish of 1 more threads
    2020-07-28 17:09:44,847 INFO: worker thread finished; awaiting finish of 0 more threads
    2020-07-28 17:09:44,848 INFO: EPOCH - 5 : training on 8191447 raw words (7021693 effective words) took 4.4s, 1610955 effective words/s
    2020-07-28 17:09:44,853 INFO: training on a 40957235 raw words (35110486 effective words) took 22.4s, 1564918 effective words/s
    2020-07-28 17:09:44,854 INFO: precomputing L2-norms of word weight vectors
    2020-07-28 17:09:44,861 INFO: saving Word2Vec object under ./word2vec.bin, separately None
    2020-07-28 17:09:44,862 INFO: not storing attribute vectors_norm
    2020-07-28 17:09:44,866 INFO: not storing attribute cum_table
    2020-07-28 17:09:44,922 INFO: saved ./word2vec.bin
    


```python
# load model
model = Word2Vec.load("./word2vec.bin")

# convert format
model.wv.save_word2vec_format('./word2vec.txt', binary=False)
```

    2020-07-28 17:09:47,602 INFO: loading Word2Vec object from ./word2vec.bin
    2020-07-28 17:09:47,639 INFO: loading wv recursively from ./word2vec.bin.wv.* with mmap=None
    2020-07-28 17:09:47,639 INFO: setting ignored attribute vectors_norm to None
    2020-07-28 17:09:47,640 INFO: loading vocabulary recursively from ./word2vec.bin.vocabulary.* with mmap=None
    2020-07-28 17:09:47,641 INFO: loading trainables recursively from ./word2vec.bin.trainables.* with mmap=None
    2020-07-28 17:09:47,642 INFO: setting ignored attribute cum_table to None
    2020-07-28 17:09:47,642 INFO: loaded ./word2vec.bin
    2020-07-28 17:09:47,653 INFO: storing 4335x100 projection weights into ./word2vec.txt
    


```python

```
