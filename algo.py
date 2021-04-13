# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
import sklearn.utils._cython_blas
import sklearn.neighbors._typedefs
import sklearn.tree
import sklearn.tree._utils
import xlsxwriter
import platform
import time
from sklearn.svm import SVR
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


path = Path('./')

if platform.system() == 'Windows':
    font = '微软雅黑'
    newline = ''
    encoding = 'utf-8'
else:
    font = 'Lucida Grande'
    newline = '\n'
    encoding = 'gbk'

def readJsonData(dataPath):
    with open(dataPath, 'r') as f:
        d = json.loads(f.read())
        
    dt = {key: [(i, float(value)) for i, value in enumerate(d[key]) if value] for key in d}
    return dt
# def readData(dataPath):
#     try:
#         with open(dataPath, 'r', encoding='gbk') as f:
#             rawdata = f.readlines()
#     except UnicodeDecodeError:
#         with open(dataPath, 'r', encoding='utf-8') as f:
#             rawdata = f.readlines()
#     if '\n' in rawdata:
#         rawdata.remove('\n')
#     rawdata = [rd.replace('\n', '') for rd in rawdata]
#     rawdata = [r[0:] for r in rawdata if r][2:]
#
#     test_data = {}
#     header = list(filter(None, rawdata[0].split('\t')))
#     for h in header:
#         test_data[h] = {}
#
#     hd = rawdata[0].split('\t')
#     print('header', hd)
#     counts = []
#
#     c = 0
#     for e in hd:
#         if e == '':
#             c += 1
#         else:
#             counts.append(c)
#             c = 0
#     counts.append(c)
#     counts = counts[1:]
#
#     indices = []
#     count = 0
#     for c in counts:
#         count = count + c + 1
#         indices.append(count)
#
#     indices = [0] + indices
#
#     i = 0
#     ind = 0
#     for d in test_data:
#         print(d)
#         features = rawdata[1].split('\t')[indices[i]:indices[i+1]]
#         print(features)
#         feas = []
#         for feaInd, feature in enumerate(features):
#             if not feature:
#                 fea = features[feaInd-1] + '%'
#                 feas.append(fea)
#             else:
#                 feas.append(feature)
#         features = feas
#
#         for feature in features:
#             values = [row.split('\t')[ind].strip() for row in rawdata[4:]]
#             try:
#                 vector = [(j, float(v.replace("%", '').replace(',', ''))) for j, v in enumerate(values)]
#             except ValueError:
#                 vector = []
#                 for j, v in enumerate(values):
#                     if v == '':
#                         ...
#                     else:
#                         try:
#                             vector.append((j, float(v.replace('%', '').replace(',', ''))))
#                         except ValueError:
#                             ...
#             test_data[d][feature] = vector
#             ind += 1
#         i += 1
#     print('test_data processed')
#
#     dates = [rawdata[i].split('\t')[0] for i in range(4, len(rawdata))]
#     logs = list(zip(dates, [rawdata[i].split('\t')[-1] for i in range(4, len(rawdata))]))
#
#     d = {'时间（过去天数）': list(zip(range(len(dates)), range(-1 * (len(dates) + 1), -1)))}
#
#     pools = {}
#     for process in test_data:
#         pools[process] = []
#         for feature in test_data[process]:
#             if test_data[process][feature]:
#                 pools[process].append(feature)
#                 d[process + '-' + feature] = test_data[process][feature]
#
#     return d, logs, pools


def knnRegress(X, maxT=None, n_points=2000, strength=0, locality=16.5):
    T = []
    for x in X:
        T += [r[0] for r in x]
    #print(T)
    T = list(set(T))
    minT = 0
    if maxT:
        maxT = maxT
    else:
        maxT = int(max(T))
    n_points = min(int(maxT-minT), n_points)    
    #print(type(n_points))
    Ts = [int(v) for v in np.linspace(minT, maxT, n_points+1)]
    trX = []
    for t in Ts:
        trx = []
        for x in X:
            ts = (1/(np.abs((np.array([r[0] for r in x]) - t)+1e-4)))**2
            ps = ts/np.sum(ts)
            xs = np.array([r[1] for r in x])
            rx = np.dot(xs, ps)
            trx.append(rx)
        
        trx = np.array(trx)
        for i in range(strength):
            trX.append(np.abs(np.exp(np.random.normal(np.log(trx), np.abs(np.log(trx))/locality))))
        trX.append(trx)
    
    trX = np.array(trX)
    Ts *= (strength + 1)
    # print(len(trX.T[0]))
    Ts = np.array(Ts)
    
    resX = np.array([list(zip(Ts, x)) for x in trX.T])

    # print('resX', resX.shape)
    return resX


def plot(y, mode='scatter',  **kwargs):
    if mode == 'scatter':
        plt.scatter([y[0] for y in y], [y[1] for y in y], **kwargs)
    elif mode == 'line':
        plt.plot([y[0] for y in y], [y[1] for y in y], **kwargs)
    # else:
    #     print('Plotting mode not recognized')


def crossPlot(x, y, **kwargs):
    
    T = []
    for t, v in x:
        if t in [y[0] for y in y]:
            T.append(t)
    tx, ty = [x[1] for x in x if x[0] in T], [y[1] for y in y if y[0] in T]
    plt.scatter(tx, ty, **kwargs)
    return


def polyFit(x, y, deg=3, xPolicy=-70, yPolicy=70, partitions=10):
    T = []
    for t, v in x:
        if t in [y[0] for y in y]:
            T.append(t)
    tx, ty = [x[1] for x in x if x[0] in T], [y[1] for y in y if y[0] in T]

    if not tx:
        return None, 0, [], []

    if xPolicy == 0 and yPolicy == 0:
        TX, TY = tx, ty
    else:
        filteredPoints = []
        rawpoints = list(zip(tx, ty))

        L = (max(tx) - min(tx))/partitions
        for i in range(partitions):
            ax = min(tx) + i*L
            bx = ax + L
            S = [point for point in rawpoints if ax <= point[0] <= bx]
            if yPolicy >= 0:
                s = [point for point in S if point[1] >= np.percentile([v[1] for v in S], yPolicy)]
            else:
                s = [point for point in S if point[1] <= np.percentile([v[1] for v in S], 100+yPolicy)]
            filteredPoints += s

        L = (max(ty) - min(ty))/partitions
        for i in range(partitions):
            ay = min(ty) + i*L
            by = ay + L
            S = [point for point in rawpoints if ay <= point[1] <= by]
            if xPolicy >= 0:
                s = [point for point in S if point[0] >= np.percentile([v[0] for v in S], xPolicy)]
            else:
                s = [point for point in S if point[0] <= np.percentile([v[0] for v in S], 100+xPolicy)]
            filteredPoints += s
        TX, TY = list(zip(*filteredPoints))

    pcoefs = np.polyfit(TX, TY, min(len(T), deg))

    # print('polynomial fitting...done')
    p = np.poly1d(pcoefs)
    n = 100
    zx = [min(tx) + (max(tx)-min(tx))*i/n for i in range(n)]
    ypreds = p(zx)
    r2s = sklearn.metrics.r2_score(y_true=TY, y_pred=p(TX))
    r2s *= (1 - 1/len(T)**0.5)
    r2s = round(np.nan_to_num(r2s, copy=True, nan=0.0, posinf=None, neginf=None), 4)
    return p, r2s, zx, ypreds


def polyfitFeatureImportances(yvar, data):
    results = {}
    for xvar in data:
        # print(xvar)
        if xvar == yvar:
            continue
        x, y = data[xvar], data[yvar]
        p, r2s, zx, ypreds = polyFit(x, y)
        results[xvar] = r2s
    return results


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


def strategy(trX, thresholds=None, typeDefs=None, safety=0.7):
    if not typeDefs:
        typeDefs = [-1]*len(inputVars) + [1/len(controlVars)]*len(controlVars) + [2]*len(outputVars)
        
    typeDefs = np.array(typeDefs)
    xinds = [i for i, td in enumerate(typeDefs) if td < 0]
    yinds = [i for i, td in enumerate(typeDefs) if 0 <= td <= 1]
    xyinds = [i for i, td in enumerate(typeDefs) if td < 2]
    zinds = [i for i, td in enumerate(typeDefs) if td >= 2]
    
    X = trX[xinds, :, 1].T
    Y = trX[yinds, :, 1].T
    XY = trX[xyinds, :, 1].T
    Z = trX[zinds, :, 1].T
    
    #model = [RandomForestRegressor(criterion='mae').fit(XY, z) for z in Z.T]
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
    model = [pipe.fit(XY, z) for z in Z.T]
    
    T = lambda xy: np.array([regr.predict(xy) for regr in model]).T
    
    THR = np.array([[[float(thresh.replace('q', '').replace("%", ''))/100, threshold[thresh]] for thresh in threshold] for threshold in thresholds])

    X1 = []
    Y1 = []
    
    def J(Y_var):
        return np.matmul(Y_var/np.mean(Y, axis=0), typeDefs[yinds])
    
    def R(Z_var):
        return np.matmul(np.array([[np.max([THR[j][i][0]*np.clip(z[j] - THR[j][i][1], 0, np.inf) for i, thresh in enumerate(threshold)]) for j, threshold in enumerate(thresholds)] for z in Z_var]) / np.mean(Z, axis=0), typeDefs[zinds])
    
    def L(X, Y, safety):
        return (1 - safety) * J(Y) + safety * R(T([[*(X[i][xinds]), *y] for i, y in enumerate(Y)])) + 0.0001
    
    #lossModel = RandomForestRegressor(criterion='mae').fit([[*(X[i][xinds]), *y] for i, y in enumerate(Y)], L(X, Y, safety=safety))
    
    
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
    lossModel = pipe.fit([[*(X[i][xinds]), *y] for i, y in enumerate(Y)], L(X, Y, safety=safety))
    
    def localLoss(x):
        xin = [[*(np.multiply(x, np.ones(Y.shape))[i][xinds]), *y] for i, y in enumerate(Y)]
        ll = lossModel.predict(xin) + 1*np.sum(np.abs(x-X)/np.mean(X, axis=0), axis=1)
        ll = ll**10
        p = (1/ll)/np.sum(1/ll)
        policy = np.matmul(p, Y)
        return policy
    
    rinds = random.sample(range(len(X)), k=int(len(X)/10))
    
    Y1 = [localLoss(x) for x in X[rinds]]

    X1,Y1 = np.array(X[rinds]), np.array(Y1)

    # strat = [RandomForestRegressor(criterion='mae').fit(X1, y) for y in Y1.T]
    
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
    strat = [pipe.fit(X1, y) for y in Y1.T]
    
    S = lambda x: np.array([regr.predict(x) for regr in strat]).T
    
    return S, T
    

def efficacy(trX, strat, T, thresholds, typeDefs=None, startIndex=0, endIndex=None, verbose=False):
    if not typeDefs:
        typeDefs = [-1]*len(inputVars) + [1/len(controlVars)]*len(controlVars) + [2]*len(outputVars)
    typeDefs = np.array(typeDefs)
    xinds = [i for i, td in enumerate(typeDefs) if td < 0]
    yinds = [i for i, td in enumerate(typeDefs) if 0 <= td <= 1]
    xyinds = [i for i, td in enumerate(typeDefs) if td < 2]
    zinds = [i for i, td in enumerate(typeDefs) if td >= 2]
    
    X = trX[xinds, :, 1]
    Y = trX[yinds, :, 1]
    XY = trX[xyinds, :, 1]
    Z = trX[zinds, :, 1]
    
    decisions = strat(trX[xinds, :, 1].T)
    xin = [[*(X.T[i][xinds]), *decision] for i, decision in enumerate(decisions)]
    
    predZ = T(xin)
    
    THR = np.array([[[float(thresh.replace('q', '').replace("%", ''))/100, threshold[thresh]] for thresh in threshold] for threshold in thresholds])
    
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
            tol = 0.005*target
            iterations = 100
            for iteration in range(iterations):
                ps = np.array([np.clip(np.random.normal(p, p/10), 0, 100) for offspring in range(3)] + [p])
                qs = np.percentile(z[startIndex:endIndex], ps)
                Ls = np.abs(qs-target)
                p = ps[np.argmin(Ls)]
                q = qs[np.argmin(Ls)]
                e = Ls[np.argmin(Ls)]
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
            tol = 0.005*target
            iterations = 100
            for iteration in range(iterations):
                ps = np.array([np.clip(np.random.normal(p, p/10), 0, 100) for offspring in range(3)] + [p])
                qs = np.percentile(Z[i][startIndex:endIndex], ps)
                Ls = np.abs(qs-target)
                p = ps[np.argmin(Ls)]
                q = qs[np.argmin(Ls)]
                e = Ls[np.argmin(Ls)]
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

    # if verbose:
    #     print('CONSUMPTIONS\n', consumptions)
    #     print('RISKS\n', risks)
    
    return decisions, predZ, consumptions, risks


def GCRF(data, inputVars, controlVars, outputVars, partitions, thresholds, typeDefs, strength=0, locality=100, errorAdjust=0):
    # print('GCRF executed')
    
    X = [data[v] for v in inputVars + controlVars + outputVars]
    trX = knnRegress(X, n_points=2000, strength=strength, locality=locality)

    workbook = xlsxwriter.Workbook('output.xlsx')
    worksheet = workbook.add_worksheet('模型输出')

    worksheet.set_column('A:A', 20)

    bold = workbook.add_format({'bold': True})
    cell_format = workbook.add_format()

    cell_format.set_align('right')
    cell_format.set_text_wrap()
    
    regionsX = [[]]
    partitions.reverse()
    for partition in partitions:
        ps = []
        for i, v in enumerate(partition):
            # print(i, v)
            if i < len(partition)-1:
                p = [partition[i], partition[i+1]]
                # print(p)
                prow = [[p] + region for region in regionsX]
                # print('prow', prow)
                ps.append(prow)
        regionsX = []
        for p in ps:
            regionsX += p

    regions = regionsX
    # print('regions', regions)
    headers = [*(inputVars + controlVars), 'p', 'n']
    result = crf(trX, [[0, np.inf]*len(trX)], typeDefs=typeDefs, thresholds=thresholds, errorAdjust=errorAdjust)
    ks = list(result['stats'][0].keys())
    headers1 = [[outputVar + '-mean'] + ks[1:] for outputVar in outputVars]
    for header in headers1:
        headers += [*header]
    headers += ['Risk', 'Loss']
    for j, header in enumerate(headers):
        worksheet.write(0, j, header, bold)
    dresult = []
    for i, region in enumerate(regions):
        for outputVar in outputVars:
            region.append([0, np.inf])
        # print(region)
        result = crf(trX, region, typeDefs=typeDefs, thresholds=thresholds, errorAdjust=errorAdjust)
        n = trX.shape[1]
        p = result['p']
        row = [*region[:-len(outputVars)], p, round(p*n)]
        for outInd, outputVar in enumerate(outputVars):
            # print('outInd', outInd, outputVar)
            # print('output Index', -(len(outputVars)-outInd))
            row +=[*list(result['stats'][-(len(outputVars)-outInd)].values())]
        row.append(round(result['Risk'], 6))
        row.append(round(result['Loss'], 6))
        for j in range(len(row)):
            worksheet.write(i+1, j, round(row[j], 8) if is_number(row[j]) else str(row[j]), cell_format)
        dresult.append(result)
    workbook.close()
    return dresult, trX


class analysis:
    def __init__(self, data, inputVars, controlVars, outputVars, thresholds, typeDefs=None, safety=0.975, startIndex=None, endIndex=None, verbose=True):
        # startIndex 和 endIndex 必须都要为负数
        xinds = [i for i, td in enumerate(typeDefs) if td < 0]
        yinds = [i for i, td in enumerate(typeDefs) if 0 <= td <= 1]
        xyinds = [i for i, td in enumerate(typeDefs) if td < 2]
        zinds = [i for i, td in enumerate(typeDefs) if td >= 2]
        maxT = np.max(np.array([data[feature] for feature in data][0])[:,0])
        txs = [data[feature] for feature in inputVars + controlVars + outputVars]
        self.trX = knnRegress(txs, maxT=maxT, strength=0)
        
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

        self.train()
        
    def train(self):  # opt 机器
        # strat = [RandomForestRegressor(criterion='mae').fit(self.X, y) for y in self.Y.T]
        
        strat = [KNeighborsRegressor().fit(self.X, y) for y in self.Y.T]
        
        self.S_hm = lambda x: np.array([regr.predict(x) for regr in strat]).T
        
        
        
        self.S, self.T = strategy(self.trX, thresholds=self.thresholds, typeDefs=self.typeDefs, safety=self.safety)
        self.Yopt, self.Zopt, consumptions, risks = efficacy(self.trX, strat=self.S, T=self.T, typeDefs=self.typeDefs, thresholds=self.thresholds, startIndex=self.startIndex, endIndex=self.endIndex, verbose=self.verbose)
        
        self.consumptions = dict(zip(self.controlVars, consumptions))
        self.risks = dict(zip(self.outputVars, risks))
        
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


if __name__ == '__main__':
    plt.style.use('dark_background')
    
    data = readJsonData(path / 'JunoProject' / 'Sheng-3.1' / 'value.json')
    
    # inputVars = ['二沉池混合后-TP', '二沉池混合后-SS']
    # controlVars = ['高效澄清池-PAC（投加量）', '高效澄清池-PAM(投加量）']
    # outputVars = ['排放池-TP在线', '高效澄清池-SS']
    
    inputVars = ['二沉池混合后-TOC (mg/L)', '缺氧池B（D-N）-NO3-N (mg/L)']
    controlVars = ['高效澄清池-粉炭(投加量) (mg/L)']
    outputVars = ['高效澄清池-TOC (mg/L)']
    # 人工定义  td = [-1, -1, -1, -1, 0.2, 0.3, 0.5, 4, 6] (情况指标全为-1，运行指标全为小数，结果指标全部大于1，各类指标内占比比例不变)
    typeDefs = [-1]*len(inputVars) + [1/len(controlVars)]*len(controlVars) + [2]*len(outputVars)
    # typeDefs = [-1,-1,1, 2]
    
    t1 = time.time()
    # boundaries = range(11, 19)
    # As = []
    # for boundary in boundaries:
    #     # thresholds = [{'q99%': 0.35, 'q80%': 0.2}, {'q99%': 9, 'q80%': 7}]

    #     # thresholkds= [{q100: 20, q90: 10}, {q95: 40, q85:30}]  (String)
    #     thresholds = [{'q99%': 19, 'q80%': boundary}]

    #     A = analysis(data, inputVars, controlVars, outputVars, thresholds, typeDefs, safety=0, verbose=True)
    #     print(A.risks)
    #     print(A.consumptions)
    #     print('time taken', time.time()-t1)
        
    #     
        
    #     As.append(A)
    #     # rx = [[0.4, 1.0], [0, np.inf]]
    #     # print('region x', dict(zip(inputVars, rx)))
    #     # B = A.crf(rx=rx)
    #     # print('P(xyz in region x)', B.p)
    # print('time taken', time.time()-t1)
    
    # Rs = [A.risks for A in As]
    # Cs = [A.consumptions for A in As]
    # # print(Rs)
    # # print(Cs)
    
    thresholds = [{'q99%': 19, 'q80%': 10}]
    A = analysis(data, inputVars, controlVars, outputVars, thresholds, typeDefs, safety=0.5, verbose=True)
    print(A.risks)
    print(A.consumptions)
    print('time taken', time.time()-t1)