# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:28:41 2023

@author: 11033
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score

import geatpy as ea
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
'''
开始
'''
np.random.seed(0)

#====================================空气压差40组=====================================
# dataset_path = './空气压差_40组.xlsx'
#====================================氢气压差40组=====================================
dataset_path = './氮气最大的P速度_40组.xlsx'
#[微通道宽度，温差和长度]H	DeltaT	L
input_list = ['H', 'DeltaT', 'L','T0','lambda']
#压升
output_list = ['u']
df = pd.read_excel(dataset_path)
inputs = df[input_list].to_numpy()
outputs = df[output_list].to_numpy().reshape(-1,)
#尺寸校正
inputs[:,[0,2,4]] = 1e-6 * inputs[:,[0,2,4]]
#长度和温度
dim_matrix = np.array(
    [
        [1., 0., 1.,0,1],
        [0., 1., 0.,1,0]
    ],
)

dim = 1
deg = 1
var_num = 5
def getcoef(inputs,outputs,GaOpt,deg,dim):
    theta = np.exp(np.log(inputs) @ np.reshape(GaOpt, [inputs.shape[1], dim]))
    poly = PolynomialFeatures(deg)
    X_poly = poly.fit_transform(theta.reshape(-1, 1))
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_poly, outputs)
    y_pred = reg.predict(X_poly)
    r2 = r2_score(outputs, y_pred)
    print("R2=",r2)
    return reg.coef_

# print('****************start find dimensionless*************')
# @ea.Problem.single
# def evalVars(x):
#     #训练一个线性模型
#     theta = np.exp(np.log(inputs) @ np.reshape(x, [inputs.shape[1], dim]))
#     poly = PolynomialFeatures(deg)
#     X_poly = poly.fit_transform(theta.reshape(-1, 1))
#     reg = LinearRegression(fit_intercept=False)
#     reg.fit(X_poly, outputs)
#     y_pred = reg.predict(X_poly)
#     ObjV = r2_score(outputs, y_pred)
#     # ObjV = np.max(abs(y_pred-outputs)/outputs)
    
#     #约束
#     CV = np.abs((dim_matrix @ x )) #列向量
#     return ObjV,CV

# problem = ea.Problem(name='soea quick start demo',
#                         M=1,  # 目标维数
#                         maxormins=[-1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
#                         Dim = var_num,  # 决策变量维数
#                         varTypes=[1]*var_num,  # 决策变量的类型列表，0：实数；1：整数
#                         lb=[-2]*var_num,  # 决策变量下界
#                         ub=[2]*var_num,  # 决策变量上界
#                         evalVars=evalVars)
# # 构建算法
# algorithm = ea.soea_SEGA_templet(problem,
#                                     ea.Population(Encoding='RI', NIND=400),
#                                     MAXGEN=6000,  # 最大进化代数。
#                                     logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
#                                     trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
#                                     maxTrappedCount=150)  # 进化停滞计数器最大上限值。
# # 求解
# res = ea.optimize(algorithm, seed=1, verbose=True, 
#                   drawing=1, outputMsg=True, drawLog=False, 
#                   saveFlag=False, dirName='result')
# GaOpt = res['Vars']

#=====================================验证===================================    
import matplotlib.pyplot as plt
GaOpt = [0,1,-1,-1,1]
Dimention_y = np.exp(np.log(inputs) @ np.reshape(GaOpt, [var_num, 1]))
Dimention_x = outputs
plt.figure()
plt.plot(Dimention_x,Dimention_y,'ro')

coef = getcoef(inputs,outputs,GaOpt,deg,dim)
#查看系数的预测能力
theta = np.exp(np.log(inputs) @ np.reshape(GaOpt, [inputs.shape[1], dim]))
if len(coef) == 2:
    y_pred = coef[0] + theta*coef[1]
elif len(coef) == 3:
    y_pred = coef[0] + theta*coef[1] + theta**2 *coef[2] 
# 


print('最大的误差为',np.max(abs(y_pred-outputs)/outputs))

# #==========================用于origin绘图用=================================
xplot = np.linspace(0, 0.01,100)
yplot = coef[0] + xplot*coef[1]
plt.figure(figsize=(8,4))
plt.plot(theta,outputs,'ro')
plt.plot(xplot,yplot,'k')