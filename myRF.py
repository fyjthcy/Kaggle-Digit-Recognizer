#__author__ = 'Hanchuanyu' results:0.96657 
#coding:UTF-8
import re
import pandas as pd
import numpy as np
import csv as csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
gbdt=GradientBoostingRegressor(
  loss='ls'
, learning_rate=0.1
, n_estimators=150
, subsample=1
, min_samples_split=2
, min_samples_leaf=1
, max_depth=7
, init=None
, random_state=None
, max_features=None
, alpha=0.9
, verbose=0
, max_leaf_nodes=None
, warm_start=False
)

if __name__ == '__main__':
    #读入test文件
    global test_org
    test_org =pd.read_csv('test.csv',header=0)
    ids=test_org['Id']


    # #读入train_result文件,test_result文件
    global train_df
    train_df =pd.read_csv('train_result.csv',header=-1)
    global test_df
    test_df =pd.read_csv('test_result.csv',header=-1)

    #读入result_offline_train文件,result_offline_test文件
    # global train_df
    # train_df =pd.read_csv('result_offline_train.csv',header=0)
    # global test_df
    # test_df =pd.read_csv('result_offline_test.csv',header=0)
    # ids=test_df['Id']

    train_data = train_df.values
    test_data = test_df.values

    #RandomForestClassifier进行预测
    # print 'Training...'
    # forest = RandomForestClassifier(n_estimators=200)
    # forest = forest.fit( train_data[0::,1:-1], train_data[0::,-1] )
    #
    # print 'Predicting...'
    # output = forest.predict(test_data[0::,1::]).astype(int)


    #GDBT预测
    gbdt.fit( train_data[0::,1:-1], train_data[0::,-1] )
    output=gbdt.predict(test_data[0::,1::])

    #offline
    #GDBT预测
    # gbdt.fit( train_data[0::,1:-1], train_data[0::,-1] )
    # output=gbdt.predict(test_data[0::,1:-1])
    #RF预测
    # forest = RandomForestClassifier(n_estimators=400)
    # forest = forest.fit( train_data[0::,1:-1], train_data[0::,-1] )

    # print 'Predicting...'
    # output = forest.predict(test_data[0::,1:-1]).astype(int)

    predictions_file = open("mysubmit.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","Response"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'

    # #offline
    # predictions_file = open("mysubmit_offline.csv", "wb")
    # open_file_object = csv.writer(predictions_file)
    # open_file_object.writerow(["Id","Response"])
    # open_file_object.writerows(zip(ids, output))
    # predictions_file.close()
    # print 'Done.'
