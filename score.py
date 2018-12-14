import pandas as pd
import numpy as np
from sklearn.ensemble import  RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import copy

#为了在jupyternotebook画图中显示中文
plt.rcParams['font.sans-serif']=['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

#导入数据
train_data = pd.read_csv('data/cs-training.csv')
train_data = train_data.iloc[:,1:]
train_data.info()



#缺失值处理，利用随机森林算法进行填充
mData = train_data.iloc[:,[5,0,1,2,3,4,6,7,8,9]]# 将缺失列 MonthlyIncome放在最前面,为了后面预测值得取值方便，并且重新赋予一个新的dataframe 

#MonthlyIncome 非空
train_known = mData[mData.MonthlyIncome.notnull()].as_matrix()

#MonthlyIncome 缺失
train_unknown = mData[mData.MonthlyIncome.isnull()].as_matrix()


# 利用随机森林预测缺失值
train_X = train_known[:,1:]
train_y = train_known[:,0]   # 0 列为MonthlyIncome缺失列

rfr=RandomForestRegressor(random_state=0,n_estimators=200,max_depth=3,n_jobs=-1)
rfr.fit(train_X,train_y)

predicted_y = rfr.predict(train_unknown[:,1:]).round(0)  #利用预测值填充原始数据中的缺失值
train_data.loc[train_data.MonthlyIncome.isnull(),'MonthlyIncome'] = predicted_y

train_data = train_data.dropna()  #舍弃缺失值
train_data = train_data.drop_duplicates()  #舍弃重复值




##异常值处理

# 查某些属性的异常值
train_box = train_data.iloc[:,[3,7,9]]
train_box.boxplot()

#去除异常值
train_data=train_data[train_data['NumberOfTime30-59DaysPastDueNotWorse']<90]
train_data=train_data[train_data['age']>0]



#切分数据集
from sklearn.cross_validation import train_test_split
X=train_data.iloc[:,1:]
y=train_data.iloc[:,0]

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=0)

ntrain_data=pd.concat([train_y,train_X],axis=1)
ntest_data = pd.concat([test_y,test_X],axis=1)



#单调分箱
# Y 中 1表示好样本
def mono_bin(Y, X, n=10):
    r = 0
    good=Y.sum()
    bad=Y.count()-good
    while np.abs(r) < 1: 
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)  
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['rate']/good)/((1-d3['rate'])/bad))
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
    woe=list(d4['woe'].round(3))
    cut=[]
    cut.append(float('-inf'))
    for i in range(1,n+1):
        qua=X.quantile(i/(n+1))
        cut.append(round(qua,4))
    cut.append(float('inf'))
    return d4,iv,cut,woe


#对几个可分箱的属性变量进行分箱操作
x1_d,x1_iv,x1_cut,x1_woe = mono_bin(train_y,train_X.RevolvingUtilizationOfUnsecuredLines)
 
x2_d,x2_iv,x2_cut,x2_woe = mono_bin(train_y,train_X.age)
 
x4_d,x4_iv,x4_cut,x4_woe = mono_bin(train_y,train_X.DebtRatio)
 
x5_d,x5_iv,x5_cut,x5_woe = mono_bin(train_y,train_X.MonthlyIncome)



#对几个有关次数值得属性进行分箱操作--- 离散值分箱
def woe_value(d1):
    d2 = d1.groupby('Bucket', as_index = True)
    good=train_y.sum()
    bad=train_y.count()-good
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate']/good)/((1-d3['rate'])/bad))
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
    woe=list(d4['woe'].round(3))
    return d4,iv,woe


#特征分箱处理
d3 = pd.DataFrame({"X": train_X['NumberOfTime30-59DaysPastDueNotWorse'], "Y": train_y})
d3['Bucket'] = d3['X']
d3_x1 = d3.loc[(d3['Bucket']<=0)]
d3_x1.loc[:,'Bucket']="(-inf,0]"
 
d3_x2 = d3.loc[(d3['Bucket']>0) & (d3['Bucket']<= 1)]
d3_x2.loc[:,'Bucket'] = "(0,1]"
 
 
d3_x3 = d3.loc[(d3['Bucket']>1) & (d3['Bucket']<= 3)]
d3_x3.loc[:,'Bucket'] = "(1,3]"
 
 
d3_x4 = d3.loc[(d3['Bucket']>3) & (d3['Bucket']<= 5)]
d3_x4.loc[:,'Bucket'] = "(3,5]"
 
 
d3_x5 = d3.loc[(d3['Bucket']>5)]
d3_x5.loc[:,'Bucket']="(5,+inf)"
d3 = pd.concat([d3_x1,d3_x2,d3_x3,d3_x4,d3_x5])
 
 
x3_d,x3_iv,x3_woe= woe_value(d3)
x3_cut = [float('-inf'),0,1,3,5,float('+inf')]



d6 = pd.DataFrame({"X": train_X['NumberOfOpenCreditLinesAndLoans'], "Y": train_y})
d6['Bucket'] = d6['X']

d6_x1 = d6.loc[(d6['Bucket']<=0)]
d6_x1.loc[:,'Bucket']="(-inf,0]"
 
 
d6_x2 = d6.loc[(d6['Bucket']>0) & (d6['Bucket']<= 1)]
d6_x2.loc[:,'Bucket'] = "(0,1]"
 
 
d6_x3 = d6.loc[(d6['Bucket']>1) & (d6['Bucket']<= 3)]
d6_x3.loc[:,'Bucket'] = "(1,3]"
 
 
d6_x4 = d6.loc[(d6['Bucket']>3) & (d6['Bucket']<= 5)]
d6_x4.loc[:,'Bucket'] = "(3,5]"
 
 
d6_x5 = d6.loc[(d6['Bucket']>5)]
d6_x5.loc[:,'Bucket']="(5,+inf)"
d6 = pd.concat([d6_x1,d6_x2,d6_x3,d6_x4,d6_x5])
 
 
x6_d,x6_iv,x6_woe= woe_value(d6)
x6_cut = [float('-inf'),1,2,3,5,float('+inf')]



d7 = pd.DataFrame({"X": train_X['NumberOfTimes90DaysLate'], "Y": train_y})
d7['Bucket'] = d7['X']
d7_x1 = d7.loc[(d7['Bucket']<=0)]
d7_x1.loc[:,'Bucket']="(-inf,0]"
 
 
d7_x2 = d7.loc[(d7['Bucket']>0) & (d7['Bucket']<= 1)]
d7_x2.loc[:,'Bucket'] = "(0,1]"
 
 
d7_x3 = d7.loc[(d7['Bucket']>1) & (d7['Bucket']<= 3)]
d7_x3.loc[:,'Bucket'] = "(1,3]"
 
 
d7_x4 = d7.loc[(d7['Bucket']>3) & (d7['Bucket']<= 5)]
d7_x4.loc[:,'Bucket'] = "(3,5]"
 
 
d7_x5 = d7.loc[(d7['Bucket']>5)]
d7_x5.loc[:,'Bucket']="(5,+inf)"
d7 = pd.concat([d7_x1,d7_x2,d7_x3,d7_x4,d7_x5])
 
 
x7_d,x7_iv,x7_woe= woe_value(d7)
x7_cut = [float('-inf'),0,1,3,5,float('+inf')]


d8 = pd.DataFrame({"X": train_X['NumberRealEstateLoansOrLines'], "Y": train_y})
d8['Bucket'] = d8['X']
d8_x1 = d8.loc[(d8['Bucket']<=0)]
d8_x1.loc[:,'Bucket']="(-inf,0]"
 
 
d8_x2 = d8.loc[(d8['Bucket']>0) & (d8['Bucket']<= 1)]
d8_x2.loc[:,'Bucket'] = "(0,1]"
 
 
d8_x3 = d8.loc[(d8['Bucket']>1) & (d8['Bucket']<= 3)]
d8_x3.loc[:,'Bucket'] = "(1,2]"
 
 
d8_x4 = d8.loc[(d8['Bucket']>2) & (d8['Bucket']<= 3)]
d8_x4.loc[:,'Bucket'] = "(2,3]"
 
 
d8_x5 = d8.loc[(d8['Bucket']>3)]
d8_x5.loc[:,'Bucket']="(3,+inf)"
d8 = pd.concat([d8_x1,d8_x2,d8_x3,d8_x4,d8_x5])
 
 
x8_d,x8_iv,x8_woe= woe_value(d8)
x8_cut = [float('-inf'),0,1,2,3,float('+inf')]



d9 = pd.DataFrame({"X": train_X['NumberOfTime60-89DaysPastDueNotWorse'], "Y": train_y})
d9['Bucket'] = d9['X']
d9_x1 = d9.loc[(d9['Bucket']<=0)]
d9_x1.loc[:,'Bucket']="(-inf,0]"
 
 
d9_x2 = d9.loc[(d9['Bucket']>0) & (d9['Bucket']<= 1)]
d9_x2.loc[:,'Bucket'] = "(0,1]"
 
 
d9_x3 = d9.loc[(d9['Bucket']>1) & (d9['Bucket']<= 3)]
d9_x3.loc[:,'Bucket'] = "(1,3]"
 
 
 
d9_x4 = d9.loc[(d9['Bucket']>3)]
d9_x4.loc[:,'Bucket']="(3,+inf)"
d9 = pd.concat([d9_x1,d9_x2,d9_x3,d9_x4])
 
 
x9_d,x9_iv,x9_woe= woe_value(d9)
x9_cut = [float('-inf'),0,1,3,float('+inf')]


d10 = pd.DataFrame({"X": train_X['NumberOfDependents'], "Y": train_y})
d10['Bucket'] = d10['X']
d10_x1 = d10.loc[(d10['Bucket']<=0)]
d10_x1.loc[:,'Bucket']="(-inf,0]"
 
 
d10_x2 = d10.loc[(d10['Bucket']>0) & (d10['Bucket']<= 1)]
d10_x2.loc[:,'Bucket'] = "(0,1]"
 
d10_x3 = d10.loc[(d10['Bucket']>1) & (d10['Bucket']<= 2)]
d10_x3.loc[:,'Bucket'] = "(1,2]" 

d10_x4 = d10.loc[(d10['Bucket']>2) & (d10['Bucket']<= 3)]
d10_x4.loc[:,'Bucket'] = "(2,3]"
 
 
d10_x5 = d10.loc[(d10['Bucket']>3) & (d10['Bucket']<= 5)]
d10_x5.loc[:,'Bucket'] = "(3,5]"
 
 
d10_x6 = d10.loc[(d10['Bucket']>5)]
d10_x6.loc[:,'Bucket']="(5,+inf)"
d10 = pd.concat([d10_x1,d10_x2,d10_x3,d10_x4,d10_x5,d10_x6])
 
 
x10_d,x10_iv,x10_woe= woe_value(d10)
x10_cut = [float('-inf'),0,1,2,3,5,float('+inf')]



#特征筛选：多变量分析---用于剔除特征高度相关的某些特征
corr = train_data.corr()
xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
yticks = list(corr.index)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='RdBu', ax=ax1, annot_kws={'size': 5,  'color': 'blue'})
ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
plt.show()


#特征筛选：单变量分析---用于剔除特征高度相关的某些特征
informationValue = []
informationValue.append(x1_iv)
informationValue.append(x2_iv)
informationValue.append(x3_iv)
informationValue.append(x4_iv)
informationValue.append(x5_iv)
informationValue.append(x6_iv)
informationValue.append(x7_iv)
informationValue.append(x8_iv)
informationValue.append(x9_iv)
informationValue.append(x10_iv)
informationValue
 
index=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
index_num = range(len(index))
ax=plt.bar(index_num,informationValue,tick_label=index)
plt.show()


#基于筛选后的特征，将特征WOE化
def trans_woe(var,var_name,x_woe,x_cut):
    woe_name = var_name + '_woe'
    for i in range(len(x_woe)):
        if i == 0:
            var.loc[(var[var_name]<=x_cut[i+1]),woe_name] = x_woe[i]
        elif (i>0) and (i<= len(x_woe)-2):
            var.loc[((var[var_name]>x_cut[i])&(var[var_name]<=x_cut[i+1])),woe_name] = x_woe[i]
        else:
            var.loc[(var[var_name]>x_cut[len(x_woe)-1]),woe_name] = x_woe[len(x_woe)-1]
    return var
 
x1_name = 'RevolvingUtilizationOfUnsecuredLines'
x2_name = 'age'
x3_name = 'NumberOfTime30-59DaysPastDueNotWorse'
x7_name = 'NumberOfTimes90DaysLate'
x9_name = 'NumberOfTime60-89DaysPastDueNotWorse'
 
train_X = trans_woe(train_X,x1_name,x1_woe,x1_cut)
train_X = trans_woe(train_X,x2_name,x2_woe,x2_cut)
train_X = trans_woe(train_X,x3_name,x3_woe,x3_cut)
train_X = trans_woe(train_X,x7_name,x7_woe,x7_cut)
train_X = trans_woe(train_X,x9_name,x9_woe,x9_cut)


#建立模型

## 对训练集结果计算AUC ROC KS值
# # 方法1： 直接用lr预测数值  -- 预测结果0.85

# test_X = trans_woe(test_X,x1_name,x1_woe,x1_cut)
# test_X = trans_woe(test_X,x2_name,x2_woe,x2_cut)
# test_X = trans_woe(test_X,x3_name,x3_woe,x3_cut)
# test_X = trans_woe(test_X,x7_name,x7_woe,x7_cut)
# test_X = trans_woe(test_X,x9_name,x9_woe,x9_cut)

# from sklearn.linear_model.logistic import LogisticRegression

# lr = LogisticRegression()
# lr.fit(train_X, train_y)

# # 注意predict 和predict_proba的区别
# resu = lr.predict_proba(test_X)
# resuLabel = lr.predict(test_X)

# print('predict_proba:',resu)
# print('-------')
# print('predict:',resuLabel)

# from sklearn.metrics import roc_curve,auc
# # X3=sm.add_constant(test_X)
# # resu=result.predict(X3)
# fpr,tpr,thershold=roc_curve(test_y,resu[:,1])
# rocauc=auc(fpr,tpr)
# plt.plot(fpr,tpr,'b',label='AUC=%0.2f'%rocauc)
# plt.legend()
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.ylabel('TPR')
# plt.xlabel('FPR')
# plt.title('逻辑回归预测')
# plt.show()

# print ('KS:',max(tpr-fpr))


#验证集测试

import statsmodels.api as sm
#为模型增加常数项，即回归线在 y 轴上的截距
X1=sm.add_constant(train_X)
logit=sm.Logit(train_y,X1)
result=logit.fit()
print(result.summary())

test_X = trans_woe(test_X,x1_name,x1_woe,x1_cut)
test_X = trans_woe(test_X,x2_name,x2_woe,x2_cut)
test_X = trans_woe(test_X,x3_name,x3_woe,x3_cut)
test_X = trans_woe(test_X,x7_name,x7_woe,x7_cut)
test_X = trans_woe(test_X,x9_name,x9_woe,x9_cut)

from sklearn.metrics import roc_curve,auc
X3=sm.add_constant(test_X)
resu=result.predict(X3)
fpr,tpr,thershold=roc_curve(test_y,resu)
rocauc=auc(fpr,tpr)
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'%rocauc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()


#建立评分卡
p = 20/np.log(2)    #设置PDO为20
q = 600 - 20*np.log(20)/np.log(2)  #设置定点分数为600 
 
def get_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores
 
#x_coe表示利用逻辑回归拟合后的系数矩阵
x_coe = [2.6084,0.6327,0.5151,0.5520,0.5747,0.4074]

baseScore = round(q + p * x_coe[0], 0)
x1_score = get_score(x_coe[1], x1_woe, p)

x1_score = get_score(x_coe[1], x1_woe, p)
x2_score = get_score(x_coe[2], x2_woe, p)
x3_score = get_score(x_coe[3], x3_woe, p)
x7_score = get_score(x_coe[4], x7_woe, p)
x9_score = get_score(x_coe[5], x9_woe, p)

score=[x1_score,x2_score,x3_score,x7_score,x9_score]




#建立一个函数使得当输入x1,x2,x3,x7,x9的值时可以返回评分数
cut_t = [x1_cut,x2_cut,x3_cut,x7_cut,x9_cut]
score =[x1_score,x2_score,x3_score,x7_score,x9_score]
def compute_score(x):  #x为数组，包含x1,x2,x3,x7和x9的取值
    tot_score = baseScore
    cut_d = copy.deepcopy(cut_t)
    print(cut_d)
    for j in range(len(cut_d)):
        cut_d[j].append(x[j])
        cut_d[j].sort()      
        print(cut_d[j])
        for i in range(len(cut_d[j])):
            if cut_d[j][i] == x[j]:
                print(score[j][i-1])  #注意一个问题，边界值重复加了两遍，是否需要？
                tot_score = score[j][i-1] +tot_score
    return tot_score


#输入一个根据筛选后的特征类似的值，并得到最后的评分结果
x_score=[0.3,44,3,3,5] 
compute_score(x_score)
