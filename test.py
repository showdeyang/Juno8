# -*- coding: utf-8 -*-
from algo import *


if __name__ == '__main__':
    # plt.style.use('dark_background')
    data = readJsonData(path / 'JunoProject' / '示例项目' / 'value.json')
    
    features = list(data.keys())
    5
    inputVars = ['二沉池混合后-TP (mg/L)', '二沉池混合后-SS (mg/L)']
    controlVars = ['高效澄清池-PAC(投加量) (mg/L)', '高效澄清池-PAM(投加量) (mg/L)']
    outputVars = ['排放池-TP在线 (mg/L)', '高效澄清池-SS (mg/L)']
    
    # inputVars = ['二沉池混合后-TOC (mg/L)', '缺氧池B（D-N）-NO3-N (mg/L)']
    # controlVars = ['高效澄清池-粉炭(投加量) (mg/L)']
    # outputVars = ['高效澄清池-TOC (mg/L)']
    
    typeDefs = [-1]*len(inputVars) + [1/len(controlVars)]*len(controlVars) + [2]*len(outputVars)
    
    
    t1 = time.time()
    
    thresholds = [{'q99%': 0.5}, {'q99%': 9, 'q80%': 7}]
    A = analysis(data, inputVars, controlVars, outputVars, thresholds, typeDefs, safety=0.5, verbose=True)
    print(A.risks)
    print(A.consumptions)
    print('time taken', time.time()-t1)
    
    ##########################
    
    trX = knnR(data, verbose=True)
    # plt.hist(A.trX[2, :,1])
    # plt.show()
    # plt.hist(trX[171,:, 1])
    # plt.show()
    res, pcs = featureImportances(trX, data, var=[])
    
    
    
