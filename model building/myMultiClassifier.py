import pandas as pd
import numpy as np


labeled_path = 'E:\\S.T.U.D.Y\\S.T.U.D.Y【4.1】\\数据挖掘\\小组大任务\\ly\\tweet_not_so_stop\\labeled_vectors.csv'

labeled=pd.read_table(labeled_path,sep=',')
n=len(labeled)#44725

vectors=labeled.iloc[:,:-2]
labels=labeled.loc[:,'label3']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.1)

y_test_list=list(y_test)

'''
from sklearn.preprocessing import label_binarize
y_train_bin=label_binarize(y_train,classes=[0,1,2,3,4])#标签转化为[[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]]这种格式

y_train_list2=np.array(list(y_train.map(lambda x:[x])))
X_train_list=np.array(X_train)
X_test_list=np.array(X_test)
'''
n_train=len(y_train)#40252
print('n_train:',n_train)
n_test=len(y_test)#4473
print('n_test:',n_test)

def accuracy(a,b):
	c=[]
	for i in range(len(a)):
		if a[i]==b[i]:
			c.append(1)
		else:
			c.append(0)
	return sum(c)/len(c)

'''
降噪前五分类：
整体NB:0.2683790965456156
整体决策树:0.24534986713906112
OneVsRest
NB:0.2843224092116918
决策树:0.2409211691762622
svm_linearsvc:0.34278122232063774
OneVsOne
决策树:
svm_linearsvc:
svm_poly:
bert:70%+


降噪后五分类：
整体NB:0.3655264922870557
整体决策树:0.30963559132573215
OneVsRest
NB:0.36329085624860274
决策树:0.2906326849988822
svm_linearsvc:0.4337133914598703
OneVsOne
NB:0.3655264922870557
决策树:0.3440643863179074
svm_linearsvc:0.4350547730829421
svm_poly:

降噪后三分类：
整体NB:0.5745584618824056
整体决策树:0.5171026156941649
//OneVsRest
NB:0.5763469707131679
决策树:0.5048066174826739
svm_linearsvc:0.6552649228705567
//OneVsOne
NB:0.5745584618824056
决策树:0.5296221775095015
svm_linearsvc:0.6568298680974737
svm_poly:0.6894701542588867

not so stop三分类：
整体NB:0.5622346368715084
整体决策树:0.49675977653631287
//OneVsRest
NB:0.5622346368715084
决策树:0.5099441340782123
svm_linearsvc:0.6511731843575419
//OneVsOne
NB:0.5622346368715084
决策树:0.5246927374301676
svm_linearsvc:0.6576536312849162
svm_poly:0.6824581005586592

'''

#-----------------以上都执行执行吧--------------

#---------------整体NB-------------------------
from sklearn.naive_bayes import GaussianNB
nb_g = GaussianNB()
result_nb_g = nb_g.fit(X_train,y_train)
predict_nb_g=result_nb_g.predict(X_test)

accuracy_nb_g=accuracy(y_test_list,predict_nb_g)#0.2683790965456156
print("accuracy_nb_g:"+str(accuracy_nb_g))

#---------------整体决策树------------------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_predict=dt.predict(X_test)

accuracy_dt=accuracy(y_test_list,dt_predict)#0.24534986713906112
print("accuracy_dt:"+str(accuracy_dt))

#--------------one-vs-all---------------------
from sklearn.multiclass import OneVsRestClassifier

#-------NB------
model_nb_g_ovr=OneVsRestClassifier(GaussianNB())
result_nb_g_ovr=model_nb_g_ovr.fit(X_train,y_train)                    
predict_nb_g_ovr=result_nb_g_ovr.predict(X_test)

accuracy_nb_g_ovr=accuracy(predict_nb_g_ovr,y_test_list)#0.2843224092116918
print("accuracy_nb_g_ovr:"+str(accuracy_nb_g_ovr))

#------决策树----
from sklearn.tree import DecisionTreeRegressor

model_tree_ovr=OneVsRestClassifier(DecisionTreeRegressor())
result_tree_ovr=model_tree_ovr.fit(X_train,y_train)
predict_tree_ovr=result_tree_ovr.predict(X_test)

accuracy_tree_ovr=accuracy(predict_tree_ovr,y_test_list)#0.2409211691762622
print("accuracy_tree_ovr:"+str(accuracy_tree_ovr))

#------LinearSVC-----
from sklearn import svm
from sklearn.svm import LinearSVC

model_svm_linearsvc_ovr=OneVsRestClassifier(LinearSVC(random_state=0,max_iter=5000))
result_svm_linearsvc_ovr=model_svm_linearsvc_ovr.fit(X_train,y_train)
predict_svm_linearsvc_ovr=result_svm_linearsvc_ovr.predict(X_test)

accuracy_svm_linearsvc_ovr=accuracy(predict_svm_linearsvc_ovr,y_test_list)#0.34278122232063774
print("accuracy_svm_linearsvc_ovr:"+str(accuracy_svm_linearsvc_ovr))

#------svm-poly------
'''
model_svm_poly_ovr=OneVsRestClassifier(svm.SVC(kernel='poly',probability=True))
result_svm_poly_ovr=model_svm_poly_ovr.fit(X_train,y_train)
predict_svm_poly_ovr=result_svm_poly_ovr.predict(X_test)

accuracy_svm_poly_ovr=accuracy(predict_svm_poly_ovr,y_test_list)#0.2843224092116918
print("accuracy_svm_linearsvc_ovr:"+str(accuracy_svm_poly_ovr))
'''

'''
#------神经元-----
import neurolab as nl

#定义一个深度神经网络，带有两个隐藏层，每个隐藏层由10个神经元组成，输出层由一个神经元组成
#newff多层前馈网络，neurolab.net.newff(minmax, size, transf=None)
#epochs：表示迭代训练的次数，show：表示终端输出的频率，lr：表示学习率

min_max=[]
for i in range(100):
	col=X_train.iloc[:,i]
	min_max.append([min(col),max(col)])
	
multilayer_net = nl.net.newff(min_max,[10,20,5])

multilayer_net.trainf = nl.train.train_gd#设置训练算法为梯度下降法

#每10个show一次，这里show10次
error = multilayer_net.train(X_train_list,y_train_bin,epochs=5000,show=1000,goal=0.01)

predicted_net=multilayer_net.sim(X_test_list)

predicted_net_labels=[np.argmax(x) for x in predicted_net]

accuracy_net=accuracy(predicted_net_labels,y_test_list)#0.18954827280779452
print("accuracy_net:"+str(accuracy_net))

'''

#K-means聚类找K

'''
#----------one-vs-all多分类-----------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

from sklearn.grid_search import GridSearchCV

result_svm=model_svm.fit(train_vectors,train_labels)
                         
predict_svm=result_svm.predict(test_vectors)
'''
'''
#--------------GridSearchCV自动调参数------------------
parameters = {
"estimator__C": [1,2,4,8],#乘法系数
"estimator__kernel": ["linear","poly","rbf","sigmod"],
#线性核函数，多项式核函数，高斯核函数，sigmod核函数
"estimator__degree":[1, 2, 3, 4],#多项式核函数阶数
}
'''
'''
parameters={
    "estimator__kernel": ["poly","rbf","sigmod"]
}

model_tunning=GridSearchCV(model, param_grid=parameters,score_func=f1_score)
model_tunning.fit(train_vectors, train_labels_bin)

print(model_tunning.best_score_)
print(model_tunning.best_params_)
'''


'''#保存模型
from joblib import dump, load
dump(model_tree, 'model_tree.joblib')
clf = load('model_tree.joblib') 
'''



#-------------------------------------------------------------------------------------------------------------
#----------one-vs-one多分类-----------------
from sklearn.multiclass import OneVsOneClassifier
#-------NB------
model_nb_g_one=OneVsOneClassifier(GaussianNB())
result_nb_g_one=model_nb_g_one.fit(X_train,y_train)                    
predict_nb_g_one=result_nb_g_one.predict(X_test)

accuracy_nb_g_one=accuracy(predict_nb_g_one,y_test_list)#0.2843224092116918
print("accuracy_nb_g_one:"+str(accuracy_nb_g_one))


from sklearn.multiclass import OneVsOneClassifier
model_tree_one=OneVsOneClassifier(DecisionTreeRegressor())

result_tree_one=model_tree_one.fit(X_train,y_train)
predict_tree_one=result_tree_one.predict(X_test)
accuracy_tree_one=accuracy(predict_tree_one,y_test_list)#0.2409211691762622
print("accuracy_tree_one:"+str(accuracy_tree_one))


from sklearn.svm import LinearSVC
model_svm_linearsvc_one=OneVsOneClassifier(LinearSVC(random_state=0,max_iter=5000))

result_svm_linearsvc_one=model_svm_linearsvc_one.fit(X_train,y_train)                    
predict_svm_linearsvc_one=result_svm_linearsvc_one.predict(X_test)
accuracy_svm_linearsvc_one=accuracy(predict_svm_linearsvc_one,list(y_test))#0.18954827280779452
print("accuracy_svm_linearsvc_one:"+str(accuracy_svm_linearsvc_one))


model_svm_poly_one=OneVsOneClassifier(svm.SVC(kernel='poly',probability=True))
result_svm_poly_one=model_svm_poly_one.fit(X_train,y_train)
predict_svm_poly_one=result_svm_poly_one.predict(X_test)

accuracy_svm_poly_one=accuracy(predict_svm_poly_one,y_test_list)#0.2843224092116918
print("accuracy_svm_poly_one:"+str(accuracy_svm_poly_one))

dump(result_svm_poly_one,'model_svm_poly_one.joblib')

'''

#----神经元----
import neurolab as nl

min_max=[]
for i in range(100):
    col=X_train.iloc[:,i]
    min_max.append([min(col),max(col)])
	
multilayer_net = nl.net.newff(min_max,[10,3])#3类

#设置训练算法为梯度下降法
multilayer_net.trainf = nl.train.train_gd

#训练网络
#epochs：表示迭代训练的次数，show：表示终端输出的频率，lr：表示学习率
train_labels_list=np.array(list(y_train.map(lambda x:[x])))
train_vectors_list=np.array(X_train)

from sklearn.preprocessing import label_binarize

train_labels_bin=label_binarize(train_labels_list,classes=[0,1,2])
#[[3],[1]]转化为[[0,0,0,1,0],[0,1,0,0,0]]的格式，所以对应的输出神经元有5个，看哪个的概率大就归为哪一类

#每10个show一次，这里show10次-----0.37
error = multilayer_net.train(train_vectors_list,train_labels_bin,epochs=100,show=10,goal=0.01)

#用训练数据运行该网络，预测结果
test_vectors_list=np.array(X_test)
predicted_net=multilayer_net.sim(test_vectors_list)

predicted_labels=[np.argmax(x) for x in predicted_net]

accuracy_net=accuracy(predicted_labels,y_test_list)
'''


