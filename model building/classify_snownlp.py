from snownlp import SnowNLP
from snownlp import sentiment
import pandas as pd
import numpy as np
import os
'''
#1. 训练模型 用自己的语料库(自带语料库是电商评论)
train_data_path = 'E://S.T.U.D.Y//S.T.U.D.Y【4.1】//数据挖掘//小组大任务//1114 疫情学习语料库//nCoV_100k_train.labled_Cleaned.csv'
train_data = pd.read_csv(train_data_path,engine='python')
train_neg = train_data.iloc[:,[3]][train_data['label']==-1]#提取列数据
train_pos = train_data.iloc[:,[3]][train_data.label==1]#提取列数据
train_neg.to_csv(r"./neg.csv",index=0,header=0)
train_pos.to_csv(r"./pos.csv",index=0,header=0)

sentiment.train('neg.csv','pos.csv')
sentiment.save('sentiment.marshal')
#sentiment.train('E:/Anaconda2/Lib/site-packages/snownlp/sentiment/neg.txt', 'E:/Anaconda2/Lib/site-packages/snownlp/sentiment/pos.txt')
#对语料库进行训练，把路径改成相应的位置
#用了默认的语料库:路径写到了sentiment模块下
'''
#2. 测试
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sentiment.marshal')
path = 'E://S.T.U.D.Y//S.T.U.D.Y【4.1】//数据挖掘//小组大任务//微博数据处理//22nCoV_100k_train.labled.csv'
test11 = pd.read_csv(path)
test11 = test11.iloc[:,:][(test11['情感倾向']=='-1')|(test11['情感倾向']=='1')]
test_review_list = [review for review in test11['微博中文内容']]
test_label_list = [label for label in test11['情感倾向']]
test_list_test=[(label,review) for label,review in list(zip(test_label_list,test_review_list)) if type(review)!=float]
senti=[SnowNLP(review).sentiments for label,review in test_list_test]

import matplotlib.pyplot as plt
x=[i for i in range(len(senti))]
plt.scatter(x,senti,c='royalblue',s=3)
plt.savefig('./测试集预测分数结果图')
plt.show()
'''
newsenti=[]#标签集
for i in senti:
    if(i>0.6):
        newsenti.append(1)
    else:
        newsenti.append(-1)
counts=0
for i in range(len(list_test)):
    if(newsenti)
'''
'''
#3. 预测
predict_path = 'E://S.T.U.D.Y//S.T.U.D.Y【4.1】//数据挖掘//小组大任务//微博数据处理//cleaned_text'
files = os.listdir(predict_path)
score=[]
for ifile in files:
    if not os.path.isdir(ifile):
        idata = pd.read_csv(path+"//"+ifile,encoding='utf-8')
        itext_list = [review for review in idata['text']]
        label_list = [label for label in idata['label']]
        list_test=[(label,review) for label,review in list(zip(label_list,review_list)) if type(review)!=float]
        senti=[SnowNLP(review).sentiments for label,review in list_test]
        ''''''
        for index, row in idata.iterrows():
            s = SnowNLP(row['text'])#分词
            score.append(s.sentiments)''''''
        newsenti=[]#标签集
        for i in senti:
            if(i>0.6):
                newsenti.append(1)
            else:
                newsenti.append(-1)
        counts=0
        for i in range(len(list_test)):
            if(newsenti[i]==list_test[i][0]):
                counts+=1'''
