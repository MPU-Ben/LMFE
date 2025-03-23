import time
from datetime import datetime

import differentMethods


start_time=time.time()
print('Start time:{}'.format(datetime.now()))


optimumAllDataPath ='../Files/F_1/optimumAllDataset_All_speices_sequences.csv'

argsForClassifier={'nFCV':5,'dataset':optimumAllDataPath,'auROC':1,'boxPlot':1,'accPlot':1,'timePlot':1}

differentMethods.runDifferentMethods(argsForClassifier)

end_time=time.time()
print('End time:{}'.format(datetime.now()))
print('Time consume:{} seconds'.format(end_time-start_time))