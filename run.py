from algo import *
from pprint import pprint

if __name__ == '__main__':
    data = readJsonData(path / 'JunoProject' / '示例项目' / 'value.json')

    trX = knnR(data, verbose=True)

    yvars = ['高效澄清池-PAC (kg/d)', '高效澄清池-PAM (kg/d)','臭氧池-臭氧产量 （kg/h）','臭氧池-功率 (kWh)' ,'高效澄清池-粉炭 (kg/d)','CBR池A-DO (mg/L)','活性污泥池A（ASR）-DO (mg/L)', '活性污泥池B（ASR）-DO (mg/L)',  'CBR池B-DO (mg/L)']
        
        
    morfi = MORFI(trX, data, yvars)
    xyz = morfi.XYZ(yvars)
    pprint(xyz)