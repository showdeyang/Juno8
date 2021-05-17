import algo
from pprint import pprint

if __name__ == '__main__':
    data = algo.readJsonData('JunoProject/港区/value.json')

    trX = algo.knnR(data, verbose=True)

    yvars = ['高效澄清池_PAC_(kg/d)', '高效澄清池_PAM_(kg/d)', '臭氧池_臭氧产量_(kg/h)', '臭氧池_功率_(kWh)', '高效澄清池_粉炭_(kg/d)', 'CBR池A_DO_(mg/L)', '活性污泥池A(ASR)_DO_(mg/L)', '活性污泥池B(ASR)_DO_(mg/L)',  'CBR池B_DO_(mg/L)']
        
        
    morfi = algo.MORFI(trX, data, yvars)
    xyz = morfi.XYZ(yvars)
    pprint(xyz)