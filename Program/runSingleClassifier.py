import time

import pandas as pd

import training
import testing

start_time=time.time()
print('开始时间：',start_time)

specie_name = 'All_species'
subfold = 'F_1'


optimumTrainDataPath='../Files/'+subfold+'/optimumTrainDataset_All_speices_sequences.csv'
optimumTestDataPath='../Files/'+subfold+'/optimumTestDataset_All_speices_sequences.csv'
optimumAllDataPath='../Files/'+subfold+'/optimumAllDataset_All_speices_sequences.csv'

print('------------训练单个分类器-----------')
choices={'LR':0,'SVM':0,'KNN':0,'DT':0,'NB':0,'Bagging':0,'RF':0,'AB':0,'GB':0,'LDA':0,'ET':0,'XGB':1,'VOTE':0,'LGBM':0,'GA-XGB':0,'BAYES-XGB':0,'SVM-XGB':0}
argsForTrain={'optimunTrainDataPath':optimumTrainDataPath,'optimunTestDataPath':optimumTestDataPath,'model':choices,'specieName':specie_name,'subfold':subfold}
training.train(argsForTrain)


print('---------评估单个分类器-----------')
argsForEval={'optimumTrainDatasetPath':optimumTrainDataPath,'optimumtestDatasetPath':optimumTestDataPath,'optimumAllDataset':optimumAllDataPath,'splited':1,'specieName':specie_name,'subfold':subfold}
testing.evaluate(argsForEval)

V_angularis='../Files/'+subfold+'/optimumAllDataset_V_angularis_sequences.csv'               #16
S_indicum = '../Files/'+subfold+'/optimumAllDataset_S_indicum_sequences.csv'                 #17
B_distachyon='../Files/'+subfold+'/optimumAllDataset_B_distachyon_sequences.csv'             #18
M_acuminate = '../Files/'+subfold+'/optimumAllDataset_M_acuminate_sequences.csv'             #19
M_polymorpha = '../Files/'+subfold+'/optimumAllDataset_M_polymorpha_sequences.csv'           #20
N_colorata = '../Files/'+subfold+'/optimumAllDataset_N_colorata_sequences.csv'               #21
specieNames = ['V. angularis','S. indicum','B. distachyon','M. acuminate','M. polymorpha','N. colorata']
speciestestDatasets = [V_angularis,S_indicum,B_distachyon,M_acuminate,M_polymorpha,N_colorata]
#######################################Other paper  DATASET###########################################

for specieData,specieName in zip(speciestestDatasets,specieNames):
    argsForEval={'optimumAllDataset':specieData,'specieName':specieName,'subfold':subfold}
    testing.evaluateByOtherSpeice(argsForEval)
argsForPlot={'subfold':subfold,'specieName':'G_max'}
# evaluateModel.plot_auc(argsForPlot)
# evaluateModel.plot_pre_recall(argsForPlot)
##################################################################################################

end_time=time.time()
print('结束时间：',end_time)
print('总耗时：',(end_time-start_time) / 60)