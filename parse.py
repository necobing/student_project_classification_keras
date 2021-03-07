import re
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib
import time
from torch.utils.data import DataLoader
import torch
import gensim
import shutil
from sys import platform

#三个映射 对应每个类别和数据集位置，方便读取
class_poetry_dir = {
    "songci": "data/songci/main.json",
    "tangshi": "data/tangshi/main.json",
    "lunyu": "data/lunyu/main.json",
    "shijing": "data/shijing/main.json",
    "sishuwujing": "data/sishuwujing/main.json"
}
#中英文类别转换
class_label = {
    "songci": 0,
    "tangshi": 1,
    "lunyu": 2,
    "shijing": 3,
    "sishuwujing": 4
}

class_map_cnlabel = {
    "0": "楚辞",
    "1": "唐诗",
    "2": "论语",
    "3": "诗经",
    "4": "四书五经"
}

def read_file():

    df_list = []
    for name, dir in class_poetry_dir.items():
        # 根据映射 把每个目录下的json文件传入
        df = pd.read_json(dir)
        # 映射名作为label加进去 比如songci
        df['label'] = name
        # 每个df拼在一起
        df_list.append(df)
    return df_list

def merge(df_list):
    # 把两个dataframe竖向拼接 相当于把所有df拼在一起
    df = pd.concat(df_list)
    return df

def split_dataset(df):
    # 分割数据集

    sentences = df['paragraphs'].values    # 首先取出paragraph列（诗经），变为numpy的数组
    # 为了方便文本分割，取掉所有中括号 对nan值迭代？
    sentences = [str(c).replace("[", "").replace("]", "").replace("\'", "").replace("nan", "") for c in sentences]
    print("================诗文格式转换================")
    print(sentences)
    # 取出label（五个类别）
    y = df['label'].values
    # sklearn的内置函数，按照0.25的比例生成训练集的sentence、测试集的sentence、训练集的标签、测试集的标签
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
    # return sentence 是切分之前的sentence 这个也是会用到的
    return sentences, y, sentences_train, sentences_test, y_train, y_test

def build_vocab(sentences):
    vocab = set() # 集合
    # cut_docs = df['paragraphs'].values.apply(lambda x: jieba.cut(x)).values
    # 用jieba做了分词
    cut_docs = [jieba.cut(c) for c in sentences]
    # 对分词后的数组进行迭代
    for doc in cut_docs:
        for word in doc:
            if word.strip():
                # 把每个词塞进去，集合是自动去重的
                vocab.add(word.strip())
    print("================词袋总数为: {n}个==================".format(n=len(vocab)))
    time.sleep(2)
    try:
        # 写入了本地
        with open('data/vocab.txt', 'w', encoding="utf8") as file:
            for word in vocab:
                file.write(word)
                file.write('\n')
    except Exception as e:
        pass


def run():
    df = read_file()
    df = merge(df)
    print("================合并dataframe完毕==================")
    time.sleep(1)
    print(df["label"].head())
    print(df["paragraphs"].head())
    # 数据集内已经比较干净，不需要去除停顿词之类的操作，有些繁体字取掉后识别会不精准
    sentences, y, sentences_train, sentences_test, y_train, y_test = split_dataset(df)
    print("================数据集分割完毕==================")
    time.sleep(1)
    print(sentences_train)
    print(y_train)
    # 创建了一个字典 传入了分割前的sentence
    build_vocab(sentences)
    print("================字典建立完毕，并写入本地==================")
    return sentences, y, sentences_train, sentences_test, y_train, y_test


# 以下是为了pytorch用的转变数据格式的代码，基本不会用到
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def return_Dataset(source_data, source_label):
    torch_data = GetLoader(source_data, source_label)
    return torch_data

def return_DataLoder(source_data, source_label):
    torch_data = return_Dataset(source_data, source_label)
    datas = DataLoader(torch_data, batch_size=6, shuffle=True, drop_last=False, num_workers=2)
    return datas

if __name__ == '__main__':
    run()

