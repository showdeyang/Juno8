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
from sklearn.linear_model import BayesianRidge, Ridge, LinearRegression
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
from numba import jit
# from multiprocessing import Pool
from sklearn.impute import KNNImputer
import time
import json
import functools
from matplotlib import pyplot as plt
import statistics
import pprint
import datetime

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
             ('2.1.3', 'efficacy added success-rate estimation loop break by tolerance. Performance increased by 5% ~ 50%.')]
         }
    d['release'] = str(datetime.datetime.now())
    d['version'] = d['changelog'][-1][0]
    d['python'] = ('.').join(map(str,sys.version_info[:3]))
    # print(d)
    if verbose:
        print('ABOUT ALGO')
        print('This is the precompiled algo.')
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

def knnRegress(X, n_points=6000, verbose=False):
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

    imputer = KNNImputer( weights='distance' ) #,n_neighbors=n_points
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
    
    THR = np.array([[[float(thresh.replace('q', '').replace("%", ''))/100, threshold[thresh]] for thresh in threshold] for threshold in thresholds], dtype=object)

    X1 = []
    Y1 = []
    
    def J(Y_var):
        return np.matmul(Y_var/np.mean(Y, axis=0), typeDefs[yinds])
    
    def R(Z_var):
        return np.matmul(np.array([[np.max([THR[j][i][0]*np.clip(z[j] - THR[j][i][1], 0, np.inf) for i, thresh in enumerate(threshold)]) for j, threshold in enumerate(thresholds)] for z in Z_var]) / np.mean(Z, axis=0), typeDefs[zinds])
    
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
        ll = lossModel.predict(xin) + 1*np.sum(np.abs(x-X)/np.mean(X, axis=0), axis=1)
        ll = np.power(ll,10)
        p = (1/ll)/np.sum(1/ll)
        policy = np.matmul(p, Y)
        return policy
    
    rinds = random.sample(range(len(X)), k=int(len(X)/10))
    
    Y1 = list(map(localLoss, X[rinds]))

    t4 = time.time()
    
    X1,Y1 = np.array(X[rinds]), np.array(Y1)
    # print(Y1[:20])
    pipe2 = Pipeline([('scaler', BarebonesStandardScaler()), ('knn', KNeighborsRegressor())])
    
    # strat = [RandomForestRegressor(n_estimators=30).fit(X1, y) for y in Y1.T]
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
    
    THR = np.array([[[float(thresh.replace('q', '').replace("%", ''))/100, threshold[thresh]] for thresh in threshold] for threshold in thresholds], dtype=object)
    
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

def knnR(data, time_dimension=True, verbose=False):
    t1 = time.time()
    print('Regressing data... This may take a while...')
    trX = knnRegress([data[feature] for feature in  data], verbose=verbose)
    
    if not time_dimension:
        trX = trX[:,:, 1]
    
    print('Regressing time:', time.time()-t1)
    print('Encoding time dimension?', time_dimension)
    print('data shape', trX.shape)
    return trX



class analysis:
    def __init__(self, data, inputVars, controlVars, outputVars, thresholds, typeDefs=None, safety=0.7, startIndex=None, endIndex=None, verbose=True):
        # startIndex 和 endIndex 必须都要为负数
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

def FIRF(trX, data):
    
    t1 = time.time()
    features = list(data.keys())
    
    
    
    ...




if __name__ == '__main__':
    plt.style.use('dark_background')
    data = readJsonData(path / 'JunoProject' / '示例项目' / 'value.json')
    
    # features = list(data.keys())
    
    # inputVars = ['二沉池混合后-TP (mg/L)', '二沉池混合后-SS (mg/L)']
    # controlVars = ['高效澄清池-PAC(投加量) (mg/L)', '高效澄清池-PAM(投加量) (mg/L)']
    # outputVars = ['排放池-TP在线 (mg/L)', '高效澄清池-SS (mg/L)']
    
    # # inputVars = ['二沉池混合后-TOC (mg/L)', '缺氧池B（D-N）-NO3-N (mg/L)']
    # # controlVars = ['高效澄清池-粉炭4(投加量) (mg/L)']
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
    res, pcs = propPCA(trX, data, var=[])
    
    
    
    
    
    
    
    
    
    
    