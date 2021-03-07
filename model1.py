import jieba
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import parse
import time

# 调用run方法得到切分好的数据集和测试集的数据以及label
sentences, y, sentences_train, sentences_test, y_train, y_test = parse.run()
# 传入label的映射
class_label = parse.class_label
time.sleep(2)
print(y_train)
print(y_test)

# 下面这两步，是因为y_train, y_test这两个值还是前面的songci tangshi这样的格式，不能用于训练，所以要对应转变为0,1这样的格式
for i in range(len(y_train)):
    if y_train[i] in class_label.keys():
        y_train[i] = class_label.get(y_train[i])
for j in range(len(y_test)):
    if y_test[j] in class_label.keys():
        y_test[j] = class_label.get(y_test[j])

print(y_train)
print(y_test)
# sentences_train = sentences_train.astype('float64')
# sentences_test = sentences_test.astype('float64')
# 转变格式，不然keras会报错，普通numpy的array格式是sklearn的模型，深度学习内要转变为张量，标签这种一维的要转变为float64
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')


class Config():
    embedding_dim = 300 # 词向量维度
    max_seq_len = 200 # 文章最大词数
    vocab_file = 'data/vocab.txt' # 词汇表文件路径
config = Config()


class Preprocessor():  # 将文本处理为向量
    def __init__(self, config):
        self.config = config
        # 初始化词和id的映射词典，预留0给padding字符，1给词表中未见过的词
        token2idx = {"[PAD]": 0, "[UNK]": 1}  # {word：id}
        with open(config.vocab_file, 'r', encoding='utf8') as reader:
            for index, line in enumerate(reader):
                token = line.strip()
                token2idx[token] = index + 2

        self.token2idx = token2idx

    # 文本数据传入后，调用transform方法，返回固定长度的词向量
    def transform(self, text_list):
        # 文本分词，并将词转换成相应的id, 最后不同长度的文本padding长统一长度，后面补0
        idx_list = [[self.token2idx.get(word.strip(), self.token2idx['[UNK]']) for word in jieba.cut(text)] for text in
                    text_list]
        idx_padding = pad_sequences(idx_list, self.config.max_seq_len, padding='post')

        return idx_padding

# textcnn建模
class TextCNN(object):
    def __init__(self, config): # 生成config方便拿上面的变量
        self.config = config
        # 传入预处理的类当中
        self.preprocessor = Preprocessor(config)
        self.class_name = {
            "0" : "楚辞",
            "1": "唐诗",
            "2" : "论语",
            "3": "诗经",
            "4" : "四书五经"
        }

    def build_model(self):
        # 模型架构搭建
        idx_input = tf.keras.layers.Input((self.config.max_seq_len,))
        # embedding词嵌入层
        input_embedding = tf.keras.layers.Embedding(len(self.preprocessor.token2idx),
                                                    self.config.embedding_dim,
                                                    input_length=self.config.max_seq_len,
                                                    mask_zero=True)(idx_input)
        convs = []
        # 并行的三个卷积层
        for kernel_size in [3, 4, 5]:
            c = tf.keras.layers.Conv1D(128, kernel_size, activation='relu')(input_embedding)
            c = tf.keras.layers.GlobalMaxPooling1D()(c)
            convs.append(c)
        fea_cnn = tf.keras.layers.Concatenate()(convs)
        # dropout剪枝
        fea_cnn_dropout = tf.keras.layers.Dropout(rate=0.4)(fea_cnn)

        # 两个全连接层 激活函数分别是relu和softmax 分为五类
        fea_dense = tf.keras.layers.Dense(128, activation='relu')(fea_cnn_dropout)
        output = tf.keras.layers.Dense(5, activation='softmax')(fea_dense)

        # 模型的编译
        model = tf.keras.Model(inputs=idx_input, outputs=output)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()

        self.model = model

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, epochs=5, batch_size=128, callbacks=None, **kwargs):
        # 训练
        self.build_model()
        # x_train训练集处理成向量形式
        x_train = self.preprocessor.transform(x_train)
        valid_data = None
        # 测试集也进行同样的操作
        if x_valid is not None and y_valid is not None:
            x_valid = self.preprocessor.transform(x_valid)
            # data和label已经打包了
            valid_data = (x_valid, y_valid)

        self.model.fit(
            x=x_train, # 训练集
            y=y_train,
            # data和label已经打包了 一并传入
            # 验证集
            validation_data=valid_data,
            # 训练集个数
            batch_size=batch_size,
            # 迭代次数
            epochs=epochs,
            # 回调函数
            callbacks=callbacks,
            **kwargs
        )

    def evaluate(self, x_test, y_test):
        # 评估
        # 传入test的文本，向量化
        x_test = self.preprocessor.transform(x_test)
        # keras预置的函数
        y_pred_probs = self.model.predict(x_test)
        # 得到预测值
        y_pred = np.argmax(y_pred_probs, axis=-1)
        # sklearn自带的方法，打印分类报告
        result = classification_report(y_test, y_pred, target_names=['唐诗', '宋词', '论语', '诗经', '四书五经'])
        print(result)

    def load_model(self, ckpt_file):
        self.build_model()
        self.model.load_weights(ckpt_file)

# 定义early stop早停回调函数
patience = 6
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

# 定义checkpoint回调函数
checkpoint_prefix = './checkpoints/textcnn_imdb_ckpt'
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    save_best_only=True)

print(y_train)
# 初始化模型类，调用类 启动训练
textcnn = TextCNN(config)
textcnn.fit(sentences_train, y_train, sentences_test, y_test, epochs=20, callbacks=[early_stop, checkpoint_callback]) # 训练

textcnn.evaluate(sentences_test, y_test) # 测试集评估
