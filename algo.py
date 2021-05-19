# -*- coding: utf-8 -*-
import numpy as np
import sys
import math
import random
from pathlib import Path
import sklearn
from sklearn import preprocessing, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge, Ridge, LinearRegression, LassoCV, RidgeCV
import sklearn.utils._cython_blas
import sklearn.neighbors._typedefs
import sklearn.tree
import sklearn.tree._utils
import platform
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
import warnings
from typing import List
# from numba import jit
# from multiprocessing import Pool
from sklearn.impute import KNNImputer
import time
import json
# import functools
from matplotlib import pyplot as plt
# import statistics
import pprint
import datetime
import pandas as pd
# from statsmodels.tsa.ar_model import AutoReg
from functools import reduce
import itertools

if not sys.warnoptions:
    warnings.simplefilter("ignore")

path = Path('./')

if platform.system() == 'Windows':
    font = '微软雅黑'
    newline = ''
    encoding = 'utf-8'
else:
    font = 'Lucida Grande'
    newline = '\n'
    encoding = 'gbk'

def about(verbose=True):
    d = {'name': 'JunoAlgo',
         'author': 'Show De Yang',
         'company': 'DataHans, Juneng',
         'changelog': [('2.0', 'Cython-precompiled algo, to improve performance. Performance increased by 40% ~ 120%'),
                       ('2.1.0', 'Added algo.about() function to display metadata about algo.pyd.'),
             ('2.1.1', 'knnRegress: n_points change from 2000 to 6000 to allow for 15 years of data. Bug will occur when n_points < n(days in data).'),
             ('2.1.2', 'pipe1 removed StandardScaler, LinearRegression added normalize=True. Performance increased by 2%.'),
             ('2.1.3', 'efficacy added success-rate estimation loop break by tolerance. Performance increased by 5% ~ 50%.'),
             ('2.2', 'Proposition Discovery Algorithm.'), 
             ('2.2.1', 'fixed some small bugs in MORFI.'),
             ('2.2.2', 'fixed typeDefs bug in MORFI.'),
             ('2.2.3', 'thresholds added str().'),
             ('2.3.0', 'now can auto-detect yvars.'),
             ('2.3.0.1', 'kNN-imputer uses weights=distance.'),
             ('2.3.0.2', 'knnR added weights=True, but uses weights=False in detectYvars.'),
             ('2.3.0.3', 'fixed some knnR bugs regarding weights and detectYvars.'),
             ('2.3.0.3.1', 'remove duplicate xvar and zvar.'),
             ('2.3.0.3.2', 'remove duplicate boundary for q99 and q80.'),
             ('2.3.0.3.3', 'localLoss added protection against NaN.'),
             ('2.3.0.3.4', 'typeDefs bug fixed.')]
         }
    d['release'] = str(datetime.datetime.now())
    d['version'] = d['changelog'][-1][0]
    d['python'] = ('.').join(map(str,sys.version_info[:3]))
    # print(d)
    if verbose:
        print('ABOUT ALGO')
        # print('This is the precompiled algo.')
        pprint.pprint(d)
    return d

about()

def readJsonData(dataPath):
    with open(dataPath, 'r') as f:
        d = json.loads(f.read())
        
    dt = {key: [[i, float(value)] for i, value in enumerate(d[key]) if value] for key in d}
    return dt

def var2ind(var, data):
    features = list(data.keys())
    return features.index(var)

def ind2var(ind, data):
    features = list(data.keys())
    return features[ind]


def g(v):
    return v[0]


def syncX(X, Ts):
    Ts = np.array(Ts)
    # print('Ts', len(Ts))
    trX = []
    for i, x in enumerate(X):
        if x:
            rx = x
            x = np.array(x)
            mt = np.setdiff1d(Ts,x[:,0]).reshape(-1,1)
            rx += np.c_[mt, np.ones(len(mt))*np.nan ].tolist()
            
        else:
            # print('empty x detected')
            rx = [[t, -1] for t in Ts]
        rx = sorted(rx, key=lambda v: v[0]) 
        
        trX.append(rx)
    
    # #length checking
    # print('length', max([len(rx) for rx in trX]))
    # errs = [(i, len(rx)) for i, rx in enumerate(trX) if len(rx) < max([len(rx) for rx in trX])]
    # print(len(errs), 'out of', len(trX))
    # print(errs)
    
    return trX

def knnRegress(X, n_points=6000, verbose=False, weights=False):
    # print(len(X))
    t1 = time.time()
    T = []
    # print(X[2][:20])
    for x in X:
        T += [r[0] for r in x]
    # print(T)
    # T = list(set(T))
    
    minT = math.floor(min(T))
    maxT = math.ceil(max(T))
    # print('maxT', maxT)
    
    n_points = min(maxT-minT, n_points)    
    
    Ts = list(set([int(v) for v in np.linspace(minT, maxT, n_points+1)]))
    
    t2 = time.time()

    trX = syncX(X, Ts)
    
    t3 = time.time()
    
    trX = np.array(trX)
    
    inX = trX[:, :, 1].T
    
    if weights:
        imputer = KNNImputer(n_neighbors=100000, weights='distance') # weights='distance'
    else:
        imputer = KNNImputer(n_neighbors=100000) # weights='distance'
    xnew = imputer.fit_transform(inX)
    xnew = xnew.T
    
    t4 = time.time()
    
    xnew = [[[Ts[i], x] for i,x in enumerate(xcol)] for xcol in xnew]
    xnew = np.array(xnew)
    
    t5 = time.time()
    if verbose:
        print('time taken in KNNRegress')
        print('finding common time', t2-t1)
        print('synchronizing time',t3-t2)
        print('knn imputing',t4-t3)
        print('reformat result', t5-t4)
    
    return xnew
    

def crf(region, trX, typeDefs, thresholds=None, errorAdjust=0):
    region = np.array(region)
    filtered = []
    for i in range(len(trX[0])):
        vector = trX[:, i, 1]
        if (region[:, 0] <= vector).all() and (vector < region[:, 1]).all():
            filtered.append(i)
    trZ = trX[:, filtered, :]
    
    adj = 1 - (100-errorAdjust)/100
    adjustment = (1-adj)
    
    result = {}
    result['p'] = trZ.shape[1]/trX.shape[1]
    result['test_data'] = trZ
    result['ind'] = trZ[:, :, 0]
    result['stats'] = []
    J = []
    R = []
    j = 0
    for i, z in enumerate(trZ):
        v = z[:, 1]
        d = {}
        if len(v) > 0:
            d['mean'] = np.mean(v)
            d['median'] = np.median(v)
            d['min'] = np.percentile(v, 0)
            d['q1%'] = np.percentile(v, 1*adjustment)
            d['q25%'] = np.percentile(v, 25*adjustment)
            d['q75%'] = np.percentile(v, 75*adjustment)
            d['q80%'] = np.percentile(v, 80*adjustment)
            d['q85%'] = np.percentile(v, 85*adjustment)
            d['q90%'] = np.percentile(v, 90*adjustment)
            d['q95%'] = np.percentile(v, 95*adjustment)
            d['q99%'] = np.percentile(v, 99*adjustment)
            d['max'] = np.percentile(v, 100*adjustment)
            d['IQR'] = np.percentile(v, 75) - np.percentile(v, 25)
            d['STDEV'] = np.std(v)
            d['MAE'] = np.mean(np.abs(v-np.mean(v)))
        else:
            d['mean'] = None
            d['median'] = None
            d['min'] = None
            d['q1%'] = None
            d['q25%'] = None
            d['q75%'] = None
            d['q80%'] = None
            d['q85%'] = None
            d['q90%'] = None
            d['q95%'] = None
            d['q99%'] = None
            d['max'] = None
            d['IQR'] = None
            d['STDEV'] = None
            d['MAE'] = None

        if 0 <= typeDefs[i] <= 1:
            if len(v) == 0:
                if region[i][1] == np.inf:
                    val = np.mean([region[i][0], np.max(trX[i, : ,1])])
                else:
                    val = np.mean(region[i])
                J.append(typeDefs[i] * val / np.mean(trX[i, :, 1]))
                
            else:
                J.append(typeDefs[i] * np.mean(v) / np.mean(trX[i, :, 1]))
        elif typeDefs[i] > 1:
            if not thresholds:
                p_over = 0
            else:
                threshold = thresholds[j]
                p_over = 0
                for thresh in threshold:
                    if not d[thresh]:
                        p_over = -100
                    elif d[thresh] > threshold[thresh]:
                        p_over = (100-float(thresh.replace('q', '').replace('%', '')))*(1+(d[thresh] - threshold[thresh]) / threshold[thresh])
                        p_over *= float(thresh.replace('q', '').replace('%', ''))/100
                    
                    R.append(typeDefs[i]*p_over)
            j += 1
        
        result['stats'].append(d)
    safety = 1/5
    L = np.mean(J) + safety*np.max(R)
    result['Loss'] = L
    result['Risk'] = safety*np.mean(R)
    
    result['points'] = [s['mean'] for s in result['stats']] + [L]
    
    return result


def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    except TypeError:
        return False
    return True

class BarebonesStandardScaler(preprocessing.StandardScaler):
    def transform(self, x):
        return (x - self.mean_) / self.var_ ** .5

class BarebonesLinearRegression(linear_model.LinearRegression):
    def predict(self, x):
        return np.matmul(x, self.coef_) + self.intercept_

def strategy(trX, thresholds=None, typeDefs=None, safety=1):
    # if not typeDefs:
    #     typeDefs = [-1]*len(inputVars) + [1/len(controlVars)]*len(controlVars) + [2]*len(outputVars)
    if not typeDefs:
        print('ERROR: typeDefs is None')
        return
    
    t0 = time.time()
    typeDefs = [float(td) for td in typeDefs]
    safety = float(safety)
    
    typeDefs = np.array(typeDefs)
    xinds = [i for i, td in enumerate(typeDefs) if td < 0]
    yinds = [i for i, td in enumerate(typeDefs) if 0 <= td <= 1]
    xyinds = [i for i, td in enumerate(typeDefs) if td < 2] 
    zinds = [i for i, td in enumerate(typeDefs) if td >= 2]
    
    X = trX[xinds, :, 1].T
    Y = trX[yinds, :, 1].T
    XY = trX[xyinds, :, 1].T
    Z = trX[zinds, :, 1].T
    
    t1 = time.time()
    
    pipe0 = Pipeline([('scaler', BarebonesStandardScaler()), ('knn', KNeighborsRegressor())])
    
    # model = [RandomForestRegressor(n_estimators=30).fit(XY, z) for z in Z.T]
    

    
    pipe0.fit(XY, Z)
    T = lambda xy: pipe0.predict(xy)
    
    t2 = time.time()
    
    THR = np.array([[[float(thresh.replace('q', '').replace("%", ''))/100, float(threshold[thresh])] for thresh in threshold] for threshold in thresholds], dtype=object)

    X1 = []
    Y1 = []
    
    def J(Y_var):
        return np.matmul(Y_var/np.mean(Y, axis=0), typeDefs[yinds])
    
    def R(Z_var):
        return np.matmul(np.array([[np.max([THR[j][i][0]*np.clip(z[j] - THR[j][i][1], 0, np.inf) for i, thresh in enumerate(threshold)]) for j, threshold in enumerate(thresholds)] for z in Z_var]) / (np.mean(Z, axis=0)+0.0000001), typeDefs[zinds])
    
    # xys = [[*(X[i]), *y] for i, y in enumerate(Y)]
    xys = np.c_[X,Y]
    
    def L(X, Y, safety):
        return (1 - safety) * J(Y) + safety * R(T(xys)) + 0.0001
   
    # pipe1 = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=3)), ('linear', BayesianRidge())])
    # pipe1 = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(normalize=True))])
    # pipe1 = Pipeline([('scaler', BarebonesStandardScaler()), ('poly', PolynomialFeatures(degree=3)), ('linear', BarebonesLinearRegression())])
    # pipe1 = Pipeline([('scaler', BarebonesStandardScaler()), ('poly', PolynomialFeatures(degree=3)), ('linear', BarebonesLinearRegression())])
    pipe1 = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', BarebonesLinearRegression(normalize=True))])
    # pipe1 = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
    # pipe1 = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='poly', degree=2))])
    # pipe1 = Pipeline([('scaler', StandardScaler()),('pca', PCA()),  ('knn', KNeighborsRegressor())])
    lossModel = pipe1.fit(xys, L(X, Y, safety=safety))
    
    t3 = time.time()
    
    def localLoss(x):
        # xin = [[*x, *y] for i, y in enumerate(Y)]
        xin = np.c_[np.repeat([x], Y.shape[0], axis=0),Y] #biggest improvement here
        ll = lossModel.predict(xin) + 1*np.sum(np.abs(x-X)/(np.mean(X, axis=0) + 1e-2), axis=1)
        ll = np.power(ll,5) + 1e-2
        p = (1/ll)/(np.sum(1/ll) + 1e-2)
        policy = np.matmul(p, Y)
        return policy
    
    rinds = random.sample(range(len(X)), k=int(len(X)/10))
    
    Y1 = list(map(localLoss, X[rinds]))

    t4 = time.time()
    
    X1,Y1 = np.array(X[rinds]), np.array(Y1)
    # print(Y1[:20])
    pipe2 = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
    
    # strat = [RandomForestRegressor(n_estimators=30).fit(X1, y) for y in Y1.T]
    
    print('X', X1.tolist())
    print('Y', Y1.tolist())
    
    pipe2.fit(X1, Y1)
    

    
    S = lambda x: pipe2.predict(x)
    
    t5 = time.time()
    
    
    print('preparing inputs', t1-t0)
    print('T-model', t2-t1)
    print('L-model', t3-t2)
    print('L2-model', t4-t3)
    print('S-model', t5-t4)
    return S, T
    

def efficacy(trX, strat, T, thresholds, typeDefs=None, startIndex=0, endIndex=None, verbose=False):
    # if not typeDefs:
    #     typeDefs = [-1]*len(inputVars) + [1/len(controlVars)]*len(controlVars) + [2]*len(outputVars)

    if not typeDefs:
        print('ERROR: typeDefs is None')
        return
    
    typeDefs = np.array(typeDefs)
    xinds = [i for i, td in enumerate(typeDefs) if td < 0]
    yinds = [i for i, td in enumerate(typeDefs) if 0 <= td <= 1]
    xyinds = [i for i, td in enumerate(typeDefs) if td < 2]
    zinds = [i for i, td in enumerate(typeDefs) if td >= 2]
    
    X = trX[xinds, :, 1]
    Y = trX[yinds, :, 1]
    Z = trX[zinds, :, 1]
    
    decisions = strat(trX[xinds, :, 1].T)
    # xin = [[*(X.T[i][xinds]), *decision] for i, decision in enumerate(decisions)]
    xin = np.c_[X.T, decisions]
    predZ = T(xin)
    
    THR = np.array([[[float(thresh.replace('q', '').replace("%", ''))/100, float(threshold[thresh])] for thresh in threshold] for threshold in thresholds], dtype=object)
    
    consumptions = []
    for i, y in enumerate(Y):
        consumption = {}
        oldUsage = np.mean(y[startIndex:endIndex])
        newUsage = np.mean(decisions[startIndex:endIndex, i])
        consumption['old_usage'] = oldUsage
        consumption['new_usage'] = newUsage
        consumption['saving'] = oldUsage - newUsage
        consumption['saving_rate'] = (oldUsage-newUsage)*100/oldUsage
        
        consumptions.append(consumption)

    risks = []
    for i, z in enumerate(predZ.T):
        risk = {}
        threshold = THR[i]
        risk['threshold'] = thresholds[i]
        actual = {thresh: np.percentile(z[startIndex:endIndex], 100*threshold[j][0]) for j, thresh in enumerate(thresholds[i])}
        risk['ai_actual'] = actual
        riskIncrements = {}
        riskP = {}
        for j, thresh in enumerate(threshold):
            p = 50
            target = thresh[1]
            tol = 0.01*target
            iterations = 100
            for iteration in range(iterations):
                ps = np.array([np.clip(np.random.normal(p, p/10), 0, 100) for offspring in range(3)] + [p])
                qs = np.percentile(z[startIndex:endIndex], ps)
                Ls = np.abs(qs-target)
                p = ps[np.argmin(Ls)]
                q = qs[np.argmin(Ls)]
                e = Ls[np.argmin(Ls)]
                if e < tol:
                    break
            key = 'q' + str(100*THR[i][j][0]) + '%'
            riskIncrements[key] = 100*thresh[0]-p
            riskP[key] = p
        
        risk['ai_risk'] = riskIncrements
        risk['ai_success_rate'] = riskP
        qs = np.array([float(v.replace('q', '').replace("%", ''))/100 for v in riskP.keys()])
        qv = np.array(list(riskP.values()))/100
        ps = qs/np.sum(qs)
        failure = (1 - np.dot(ps, qv))*100
        risk['ai_failure_rate'] = failure
        risk['hm_actual'] = {thresh: np.percentile(Z[i][startIndex:endIndex], 100*threshold[j][0]) for j, thresh in enumerate(thresholds[i])}
        
        riskIncrements = {}
        riskP = {}
        for j, thresh in enumerate(threshold):
            p = 50
            target = thresh[1]
            tol = 0.01*target
            iterations = 100
            for iteration in range(iterations):
                ps = np.array([np.clip(np.random.normal(p, p/10), 0, 100) for offspring in range(3)] + [p])
                qs = np.percentile(Z[i][startIndex:endIndex], ps)
                Ls = np.abs(qs-target)
                p = ps[np.argmin(Ls)]
                q = qs[np.argmin(Ls)]
                e = Ls[np.argmin(Ls)]
                if e < tol:
                    break
            key = 'q' + str(100*THR[i][j][0]) + '%'
            riskIncrements[key] = 100*thresh[0]-p
            riskP[key] = p
            
        risk['hm_risk'] = riskIncrements
        risk['hm_success_rate'] = riskP
        qs = np.array([float(v.replace('q', '').replace("%", ''))/100 for v in riskP.keys()])
        qv = np.array(list(riskP.values()))/100
        ps = qs/np.sum(qs)
        failure = (1 - np.dot(ps, qv))*100
        risk['hm_failure_rate'] = failure
        risk['ai_advantage'] = (risk['hm_failure_rate'] - risk['ai_failure_rate'])*max(risk['hm_failure_rate'], risk['ai_failure_rate'])/(risk['hm_failure_rate'] + risk['ai_failure_rate']+1)
        risks.append(risk)
    
    return decisions, predZ, consumptions, risks

def knnR(data, time_dimension=True, verbose=False, weights=False):
    t1 = time.time()
    print('Regressing data... This may take a while...')
    trX = knnRegress([data[feature] for feature in  data], verbose=verbose, weights=weights)
    
    if not time_dimension:
        trX = trX[:,:, 1]
    
    print('Regressing time:', time.time()-t1)
    print('Encoding time dimension?', time_dimension)
    print('data shape', trX.shape)
    return trX



class analysis:
    def __init__(self, data, inputVars, controlVars, outputVars, thresholds, typeDefs=None, safety=0.7, startIndex=None, endIndex=None, verbose=True):
        # startIndex 和 endIndex 必须都要为负数
        
        typeDefs = [float(td) for td in typeDefs]
        safety = float(safety)
        
        print('ELAPSE TIME BREAKDOWN')
        t1 = time.time()
        xinds = [i for i, td in enumerate(typeDefs) if td < 0]
        yinds = [i for i, td in enumerate(typeDefs) if 0 <= td <= 1]
        # xyinds = [i for i, td in enumerate(typeDefs) if td < 2]
        zinds = [i for i, td in enumerate(typeDefs) if td >= 2]
        
        txs = [data[feature] for feature in inputVars + controlVars + outputVars]
        #maxT = np.max(np.array([data[feature] for feature in data][0])[:, 0])
        self.trX = knnRegress(txs)
        t2 = time.time()
        X = self.trX[xinds, :, 1].T
        Y = self.trX[yinds, :, 1].T
        Z = self.trX[zinds, :, 1].T
        
        self.X = X
        self.Y = Y  # 历史：人
        self.Z = Z
        
        self.data = data
        self.inputVars = inputVars
        self.controlVars = controlVars
        self.outputVars = outputVars
        self.thresholds = thresholds
        self.typeDefs = typeDefs
        self.safety = safety
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.verbose = verbose
        self.xinds = xinds
        self.yinds = yinds
        self.zinds = zinds
        
        self.Xhm = self.X[startIndex:endIndex] 
        self.Yhm = self.Y[startIndex:endIndex] 
        self.Zhm = self.Z[startIndex:endIndex] 
        self.Xopt = self.Xhm
        t3 = time.time()
        self.train()
        t4 = time.time()
        print('knn regress time', t2-t1)
        print('variables declaration time', t3-t2)
        print('——'*40)
        print('TOTAL TIME TAKEN', t4-t1,)
        print('——'*40)
        
    def train(self):  # opt 机器
        t5 = time.time()
        pipe3 = Pipeline([('scaler', BarebonesStandardScaler()), ('knn', KNeighborsRegressor())])
        # strat = [RandomForestRegressor(n_estimators=30).fit(self.X, y) for y in self.Y.T]
        strat = pipe3.fit(self.X, self.Y)
        
        self.S_hm = lambda x: strat.predict(x)
        t6 = time.time()
        self.S, self.T = strategy(self.trX, thresholds=self.thresholds, typeDefs=self.typeDefs, safety=self.safety)
        t7 = time.time()
        self.Yopt, self.Zopt, consumptions, risks = efficacy(self.trX, strat=self.S, T=self.T, typeDefs=self.typeDefs, thresholds=self.thresholds, startIndex=self.startIndex, endIndex=self.endIndex, verbose=self.verbose)
        
        self.consumptions = dict(zip(self.controlVars, consumptions))
        self.risks = dict(zip(self.outputVars, risks))
        t8 = time.time()
        print('Shm-Model', t6-t5)
        # print('strategy time', t7-t6)
        print('efficacy time', t8-t7)
    def crf(self, rx=None, ry=None, rz=None):
        if not rx:
            rx = [[0, np.inf]]*len(self.xinds)
        if not ry:
            ry = [[0, np.inf]]*len(self.yinds)
        if not rz:
            rz = [[0, np.inf]]*len(self.zinds)
            
        region = rx + ry + rz
        
        result = crf(region, trX=self.trX, typeDefs=self.typeDefs, thresholds=self.thresholds)
        
        class crfRes():
            def __init__(self, result):
                self.stats = result['stats']
                self.p = result['p']
                self.indices = result['ind'][0]
        
        res = crfRes(result)
        return res



def propPCA(trX, data, var=None, n_components=None):
    t1 = time.time()
    features = list(data.keys())
    # ind = var2ind(var, data)
    
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA())]).fit(trX[:, :, 1].T)
    # strat = [RandomForestRegressor(n_estimators=30).fit(self.X, y) for y in self.Y.T]
    pca = pipe[1]
    vrs = pca.explained_variance_ratio_ #85 most important principle components such that sum(vrs) > 0.95
    
    thr = np.percentile(vrs, 95)
    c = len(vrs[vrs>=thr])
    print('components', c)
    pcs = np.abs(pca.components_)[:c] #principle components
    thresh = np.percentile(pcs, 95)
    res = []
    
    for pc in pcs:
        
        fis = list(zip(features, pc))
        
        feas = list(sorted(filter(lambda x: x[1] > thresh, fis), key=lambda x: x[1], reverse=True))
        
        res.append(feas) 
    
    res = list(filter(lambda x: len(x)>=3, res))
    
    result = []
    if var:
        for r in res:
            vs = [vr in [v[0] for v in r] for vr in var]
            if np.array(vs).all() > 0:
                result.append(r)
    else:
        result = res
    
    print('fi time', time.time()-t1)
    print('命题数量', len(result))
    return result, pcs

class process(object):
    def __init__(self, trX, data):
        # here inserts the algorithm for learning the process. I used a generic process as an example.
        
        features = list(data.keys())
        delimiters = ['_','-']
        for delimiter in delimiters:
            if delimiter in (',').join(features):
                break
        
        extracted =  [feature.split(delimiter)[0].strip() for feature in features]
        
        pools = []
        for pool in extracted:
            if pool not in pools:
                pools.append(pool)
        
        self.features = features
        self.delimiters = delimiters
        self.pools = pools
        
    def order(self, var):
        #var is something like '高效澄清池-TOC (mg/L)'
        for ind, pool in enumerate(self.pools):
            if pool in var:
                return ind
        
        return None
    
    def dist(self, var1, var2):
        # distance of var2 from var1
        try:
            d = self.order(var2) - self.order(var1)
        except TypeError:
            d = None
        
        return d

def taylor_series(A, power=20):
    # matrices = [(1/math.factorial(exponent))*np.linalg.matrix_power(A, exponent) for exponent in range(power)]
    matrices = [(1/math.factorial(exponent))*np.linalg.matrix_power(A, exponent) for exponent in range(1,power)]
    res = reduce(np.add, matrices)
    return res

    
class MORFI(object):
    #MultiOutput Regressor-based Feature Importances
    def __init__(self, trX, data, yvars):
        t1 = time.time()
        self.features = list(data.keys())
        
        d = trX[:,:,1].T
        lags = 1
        inX = np.array([d[i-lags:i].flatten() for i, x in enumerate(d) if i>=lags])
        inY = d[lags:,:]
        
        # inX1 = d[:-2,:]
        # inX2 = d[1:-1,:]
        # inX = inX2-inX1
        
        # # inY1 = d[1:-1,:]
        # inY2 = d[2:,:]
        # inY = inY2 
        
        
        # pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
        # pipe = Pipeline([('scaler', StandardScaler()), ('rf', MultiOutputRegressor(estimator=RidgeCV()))])
        pipe = Pipeline([('scaler', StandardScaler()),('ridge', MultiOutputRegressor(estimator=BayesianRidge(), n_jobs=-1))])
        pipe.fit(inX, inY)
        
        res = pipe.predict(inX)
        
        
        coefs = np.array([pipe[-1].estimators_[i].coef_ / np.max(np.abs(pipe[-1].estimators_[i].coef_)) for i, fea in enumerate(self.features)])
        # coefs=None
        # coefs = np.array([pipe[-1].estimators_[i].feature_importances_ for i, fea in enumerate(features)])
        
        # coefs = coefs**3 / np.sum(coefs**3,axis=1)
        
        coefs = np.nan_to_num(coefs)
        # coefs[coefs > 0.5] = 1
        
        # coefs[coefs <= -0.5] = -1
        
        
        coefsLower = np.tril(coefs)
        coefsUpper = np.triu(coefs)
        
        # print(coefs.tolist())
        # coefs = np.sum([np.linalg.matrix_power(coefs, i) for i in range(5)])
        # coefs = 1 + coefs + 0.5* np.linalg.matrix_power(coefs, 2) + (1/6)* np.linalg.matrix_power(coefs, 3) +  (1/24)* np.linalg.matrix_power(coefs, 4) +  (1/120)* np.linalg.matrix_power(coefs, 5) +  (1/720)* np.linalg.matrix_power(coefs, 6)
        
        # coefs = taylor_series(coefs, power=100)
        coefsForward =  taylor_series(coefsUpper, power=20)
        coefsBackward =  taylor_series(coefsLower, power=20)
        
        print('morfi time', time.time()-t1)
        self.data = data
        self.trX = trX
        self.X = inX
        self.Y = inY
        self.coefs = coefs
        self.coefsLower = coefsLower
        self.coefsUpper = coefsUpper 
        self.coefsForward = coefsForward
        self.coefsBackward = coefsBackward
        
        
        self.Ypred = res
        
        
    def fi(self, var, n_features=None, verbose=False, traverse=0):
        
        if traverse == 0:
            coefs = self.coefs
        elif traverse > 0:
            coefs = self.coefsForward
        elif traverse < 0:
            coefs = self.coefsBackward
        else:
            coefs = self.coefs
        
        
        ind = var2ind(var, self.data)
        # print("analyzing", var)
        
        prc = process(self.trX, self.data)
        self.prc = prc
        ds = list(map( lambda fea: self.prc.dist(var, fea),  self.features))
        orders = list(map(lambda fea: prc.order(fea), self.features))
        s = list(zip(self.features, coefs[ind], ds, orders))
        
        prc = process(self.trX, self.data)
        self.prc = prc
        # s = filter(lambda x: -3 <= x[2] <= 2 , sorted(s, key=lambda x: x[2]))
        s = sorted(s, key=lambda x: np.abs(x[1]) , reverse=True)[:n_features]
        
        if verbose:
            pprint.pprint(s)
        return s    
    
    def y2z(self, yvar, yvars, n_features=20):
        fi = self.fi(yvar, n_features=n_features, traverse=1)
        fiz = list(filter(lambda x: x[0] not in yvars, fi))
        return fiz
    
    def z2xy(self, zvar, yvar, yvars, n_features=2):
        currentOrder = self.prc.order(yvar)
        # print('currentOrder', currentOrder)
        fi = self.fi(zvar, n_features=n_features, traverse=-1)
        fi = list(filter(lambda x: x[3] < currentOrder, fi))
        
        fiy = list(filter(lambda x: x[0] in yvars, fi))
        fix = list(filter(lambda x: x[0] not in yvars, fi))
        
        
        return fix, fiy
    
    def Z2XY(self, fiz, yvar, yvars, n_features=2):
        res = list(map( lambda fz: self.z2xy(fz[0], yvar=yvar, yvars=yvars, n_features=n_features), fiz))
        
        #merging fix and fiy
        res = list(zip(*res))
        x = [list(map(lambda row: row[0], case)) for case in res[0]]
        y = [list(map(lambda row: row[0], case)) for case in res[1]]
        
        x = sorted(list(set(itertools.chain(*x))))
        y = sorted(list(set(itertools.chain(*y))))
        if yvar not in y:
            y.append(yvar)
        return x,y
    
    def xyz(self, yvar, yvars, verbose=False, nz=5, nxy=2):
        fiz = self.y2z(yvar, yvars, n_features=nz*4)
        x,y = self.Z2XY(fiz, yvar, yvars, n_features=nxy)
        z = list(map(lambda fz: fz[0], fiz[:nz]))
        for xv in x:
            if '日期' in xv or 'ate' in xv.lower() or 'ime' in xv.lower():
                x.remove(xv)
        
        for zvar in z:
            if zvar in x:
                z.remove(zvar)
        
        if verbose:
            print('x')
            pprint.pprint(x)
            print('y')
            pprint.pprint(y)
            print('z')
            pprint.pprint(z)
        
        return x,y,z

    def XYZ(self, yvars):
        props = []
        for yvar in yvars:
            
            x,y,z = self.xyz(yvar, yvars)
            thresholds = []
            for zvar in z:
                threshold = {}
                ind =  var2ind(zvar, self.data)
                
                th1 = round(np.percentile(self.trX[ind, :, 1], 70), 2)
                th2 = round(np.percentile(self.trX[ind, :, 1], 50), 2)
                
                threshold['q99%'] = str(th1)
                
                if th1 - th2 > 0.01:
                    threshold['q80%'] = str(th2)
                thresholds.append(threshold)
            
            prop = {}
            prop['name'] = '基于' + (',').join(y).replace('(mg/L)','').replace('（mg/L）','').strip() + '的命题'
            prop['inputVars'] = x
            prop['controlVars'] = y
            prop['outputVars'] = z
            
            # td = [-1]*len(x) + [1/len(y)]*len(y) + [2]*len(z)
            
            td : List[float] 
            td = []
            for v in x:
                td.append(-1)
            for v in y:
                td.append(1/float(len(y)))
            for v in z:
                td.append(2)
            
            # td = [str(v) for v in td]
            
            prop['typeDefs'] = td
            prop['safety'] = str(0.5)
            prop['thresholds'] = thresholds
            
            props.append(prop)
        return props
        
def crossPlot(varX, varY, trX, data):
    indX = var2ind(varX, data)
    indY = var2ind(varY, data)
    x = trX[indX,:,1]
    y = trX[indY,:,1]       
    plt.scatter(x,y, s=5)
    plt.xlabel(varX)
    plt.ylabel(varY)
    
def detectYvars(data, prob=False):
    trX = knnR(data, weights=True)
    features = list(data.keys())
    diff = np.diff(trX[:,:,1])
    
    diff[diff>0] = 1
    diff[diff==0] = 0
    diff[diff<0] = 1
    
    ps = 1-np.sum(diff, axis=-1)/diff.shape[-1]
    
    pyvs = list(zip(features, ps))
    
    pyvs = sorted(pyvs, key=lambda x: x[1], reverse=True)
    
    result = []
    for pyv in pyvs:
        c = 0
        for chars in ['排放','色度', '%', 'SS', '出水', '差', 'mg/L']:
            if chars in pyv[0]:
                c += 1
        if c==0:
            result.append(pyv)
    
    result = list(filter(lambda x: 1 > x[1]>=0.05, result))
    
    if prob:
        return result
    else:
        return list(list(zip(*result))[0])
    
    



if __name__ == '__main__':
    plt.style.use('dark_background')
    data = readJsonData(path / 'JunoProject_old' / '示例项目' / 'value.json')
    
    # features = list(data.keys())
    
    # inputVars = ['二沉池混合后-TP (mg/L)', '二沉池混合后-SS (mg/L)']
    # controlVars = ['高效澄清池-PAC(投加量) (mg/L)', '高效澄清池-PAM(投加量) (mg/L)']
    # outputVars = ['排放池-TP在线 (mg/L)', '高效澄清池-SS (mg/L)']
    
    # # inputVars = ['二沉池混合后-TOC (mg/L)', '缺氧池B（D-N）-NO3-N (mg/L)']
    # # controlVars = ['高效澄清池-粉炭(投加量) (mg/L)']
    # # outputVars = ['高效澄清池-TOC (mg/L)']
    
    # typeDefs = [-1]*len(inputVars) + [1/len(controlVars)]*len(controlVars) + [2]*len(outputVars)
    
    
    # t1 = time.time()
    
    # thresholds = [{'q99%': 0.5}, {'q99%': 9, 'q80%': 7}]
    # A = analysis(data, inputVars, controlVars, outputVars, thresholds, typeDefs, safety=0.5, verbose=True)
    # print(A.risks)
    # print(A.consumptions)
    # print('time taken', time.time()-t1)
    
    ##########################
    # data = readJsonData('C:/Users/showd/code/Juno8/JunoProject/江宁化工/value.json')
    trX = knnR(data, verbose=True)
    # plt.hist(A.trX[2, :,1])
    # plt.show()
    # plt.hist(trX[171,:, 1])
    # plt.show()
    # res, pcs = propPCA(trX, data, var=[])
    
    # yvars = ['高效澄清池-PAC (kg/d)', '高效澄清池-PAM (kg/d)','臭氧池-臭氧产量 （kg/h）','臭氧池-功率 (kWh)' ,'高效澄清池-粉炭 (kg/d)','CBR池A-DO (mg/L)','活性污泥池A（ASR）-DO (mg/L)', '活性污泥池B（ASR）-DO (mg/L)',  'CBR池B-DO (mg/L)']
    
    yvars = detectYvars(data)
    print(yvars)
    morfi = MORFI(trX, data, yvars)
    # fi = morfi.fi('臭氧池-电耗 (kWh/kgO3)', n_features=20, verbose=False, traverse=1)
    
    # prc = process(trX, data)
    # print(prc.pools)
    
    # crossPlot('二沉池混合后-TOC (mg/L)','高效澄清池-TOC (mg/L)' , trX, data)
    res = morfi.XYZ(yvars)
    
    As = [analysis(data, r['inputVars'], r['controlVars'], r['outputVars'], r['thresholds'], r['typeDefs'], safety=r['safety'], verbose=True) for r in res]
    
    # pyvs = detectYvars(trX,data)
    