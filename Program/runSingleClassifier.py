import time

import pandas as pd

import training
import testing

start_time=time.time()
print('start_time:',start_time)

specie_name = 'All_species'
subfold = 'F_1'


TrainDataPath='../Files/'+subfold+'/TrainDataset_All_speices_sequences.csv'
TestDataPath='../Files/'+subfold+'/TestDataset_All_speices_sequences.csv'
AllDataPath='../Files/'+subfold+'/AllDataset_All_speices_sequences.csv'

print('------------training single Classfier-----------')
choices={'LR':0,'SVM':0,'KNN':0,'DT':0,'NB':0,'Bagging':0,'RF':0,'AB':0,'GB':0,'LDA':0,'ET':0,'XGB':1,'VOTE':0,'LGBM':0,'GA-XGB':0,'BAYES-XGB':0,'SVM-XGB':0}
argsForTrain={'TrainDataPath':TrainDataPath,'TestDataPath':TestDataPath,'model':choices,'specieName':specie_name,'subfold':subfold}
training.train(argsForTrain)

argsForEval={'TrainDatasetPath':TrainDataPath,'testDatasetPath':TestDataPath,'AllDataset':AllDataPath,'splited':1,'specieName':specie_name,'subfold':subfold}
testing.evaluate(argsForEval)

V_angularis='../Files/'+subfold+'/AllDataset_V_angularis_sequences.csv'               
S_indicum = '../Files/'+subfold+'/AllDataset_S_indicum_sequences.csv'                
B_distachyon='../Files/'+subfold+'/AllDataset_B_distachyon_sequences.csv'             
M_acuminate = '../Files/'+subfold+'/AllDataset_M_acuminate_sequences.csv'             
M_polymorpha = '../Files/'+subfold+'/AllDataset_M_polymorpha_sequences.csv'           
N_colorata = '../Files/'+subfold+'/AllDataset_N_colorata_sequences.csv'               
specieNames = ['V. angularis','S. indicum','B. distachyon','M. acuminate','M. polymorpha','N. colorata']
speciestestDatasets = [V_angularis,S_indicum,B_distachyon,M_acuminate,M_polymorpha,N_colorata]


for specieData,specieName in zip(speciestestDatasets,specieNames):
    argsForEval={'AllDataset':specieData,'specieName':specieName,'subfold':subfold}
    testing.evaluateByOtherSpeice(argsForEval)
argsForPlot={'subfold':subfold,'specieName':'G_max'}
# evaluateModel.plot_auc(argsForPlot)
# evaluateModel.plot_pre_recall(argsForPlot)


end_time=time.time()
print('end_time：',end_time)
print('time_consume：',(end_time-start_time) / 60)
