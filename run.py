import algo
from pprint import pprint

if __name__ == '__main__':
    data = algo.readJsonData('JunoProject/港区/value.json')

    trX = algo.knnR(data, verbose=True)
    
    ### 所有值的拟合填充都已经在trX
    ### trX 的矩阵结构是 [指标index，天数，[天数，指标值]]。
    ## 比如，想读'二沉池B_TP_(mg/L)'的最后一天拟合填充值：
    
    # 先查找指标的index
    ind = algo.var2ind('二沉池B_TP_(mg/L)', data)
    
    # 根据index直接读取值。
    
    val = trX[ind, -1, 1] #-1读取最后一天，第三维的1读取指标值。
    
    print('二沉池B_TP_(mg/L)','最后一天拟合值:', val)
    
    # yvars = ['高效澄清池_PAC_(kg/d)', '高效澄清池_PAM_(kg/d)', '臭氧池_臭氧产量_(kg/h)', '臭氧池_功率_(kWh)', '高效澄清池_粉炭_(kg/d)', 'CBR池A_DO_(mg/L)', '活性污泥池A(ASR)_DO_(mg/L)', '活性污泥池B(ASR)_DO_(mg/L)',  'CBR池B_DO_(mg/L)']
    
    #new function to automatically detect yvars
    yvars = algo.detectYvars(data)
    
    morfi = algo.MORFI(trX, data, yvars)
    xyz = morfi.XYZ(yvars)
    pprint(xyz)