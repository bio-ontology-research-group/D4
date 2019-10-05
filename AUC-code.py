import pickle
import numpy as np
rank_stats = {}

with open('/Users/noora/Documents/CBRC2018/Neural_Network/newTest/D4paper/AUCs/GO/DDIGO.pckl','rb') as f:
    rank_stats['GO'] = pickle.load(f)
    rank_stats['GO'] =(827,rank_stats['GO'][1])
with open('/Users/noora/Documents/CBRC2018/Neural_Network/newTest/D4paper/AUCs/HPO/DDIHPO.pckl','rb') as f:
    rank_stats['HPO'] = pickle.load(f)
    rank_stats['HPO'] =(978,rank_stats['HPO'][1])
with open('/Users/noora/Documents/CBRC2018/Neural_Network/newTest/D4paper/AUCs/GOHPO/DDIGOHPO.pckl','rb') as f:
    rank_stats['GO∩HPO'] = pickle.load(f)
    rank_stats['GO∩HPO'] =(978,rank_stats['GO∩HPO'][1])
with open('/Users/noora/Documents/CBRC2018/Neural_Network/newTest/D4paper/AUCs/GOHPOALL/DDIGOHPOALL.pckl','rb') as f:
    rank_stats['GOUHPO'] = pickle.load(f)
    rank_stats['GOUHPO'] =(978,rank_stats['GOUHPO'][1])
    

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

x_dict = {'GO':[], 'HPO':[], 'GO∩HPO':[], 'GOUHPO':[]}
y_dict = {'GO':[], 'HPO':[], 'GO∩HPO':[], 'GOUHPO':[]}
max_auc = {'GO':0, 'HPO':0, 'GO∩HPO':0, 'GOUHPO':0}


    
    
#with open(r'C:\Users\USER\Desktop\python\4sets\HPO\DDIHPO.pckl','rb') as f:
 #   rank_stats['HPO'] = pickle.load(f)
  #  rank_stats['HPO'] =(848,rank_stats['HPO'][1])
#with open(r'C:\Users\USER\Desktop\python\4sets\GOHPO\DDIGOHPO.pckl','rb') as f:
 #   rank_stats['GO∩HPO'] = pickle.load(f)
  #  rank_stats['GO∩HPO'] =(697,rank_stats['GO∩HPO'][1])
#with open(r'C:\Users\USER\Desktop\python\4sets\GOHPOALL\DDIGOHPOALL.pckl','rb') as f:
 #   rank_stats['GOUHPO'] = pickle.load(f)
  #  rank_stats['GOUHPO'] =(978,rank_stats['GOUHPO'][1])

# pair biased auc
#import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
#x_dict = {'pkInducer':[], 'pkInhibtor':[], 'MoA':[], 'sideEffect':[]}
#y_dict = {'pkInducer':[], 'pkInhibtor':[], 'MoA':[], 'sideEffect':[]}
#max_auc = {'pkInducer':0, 'pkInhibtor':0, 'MoA':0, 'sideEffect':0}
for experiment in max_auc.keys():
    print(experiment)
    rank_counts = []
    epochs = 50
    for i in range(epochs):
        rank_counts.append(dict())
    for i in range(epochs):
        for patho, ranks in rank_stats[experiment][1][i].items():
            for rank in ranks:
                if rank not in rank_counts[i]:
                    rank_counts[i][rank] = 0
                rank_counts[i][rank]+=1
        auc_x = list(rank_counts[i].keys())
        auc_x.sort()
        auc_y = []
        tpr = 0
        step = 1/sum(rank_counts[i].values())
        for x in auc_x:
            tpr += rank_counts[i][x]*step
            auc_y.append(tpr)
        auc_x.append(rank_stats[experiment][0])
        auc_y.append(1)
        auc = np.trapz(auc_y, auc_x)/rank_stats[experiment][0]
        if auc > max_auc[experiment]:
            max_auc[experiment] = auc
            x_dict[experiment] = np.array(auc_x)/rank_stats[experiment][0]
            y_dict[experiment] = auc_y
            print('Rank based auc is: %f' % (auc)) 
            
fig=plt.figure(figsize=(4, 4), dpi= 200)
plt.plot(x_dict['GO'], y_dict['GO'], label = 'GO (AUC=' + '%.3f)' % max_auc['GO'])
plt.plot(x_dict['HPO'], y_dict['HPO'], label = 'HPO (AUC=' + '%.3f)' % max_auc['HPO'])
plt.plot(x_dict['GO∩HPO'], y_dict['GO∩HPO'], label = 'GO∩HPO (AUC=' + '%.3f)' % max_auc['GO∩HPO'])
plt.plot(x_dict['GOUHPO'], y_dict['GOUHPO'], label = 'GOUHPO (AUC=' + '%.3f)' % max_auc['GOUHPO'])





#plt.plot(x_dict['BiologicalProcess'], y_dict['BiologicalProcess'], label = 'BiologicalProcess (AUC=' + '%.3f)' % max_auc['BiologicalProcess'])
#plt.plot(x_dict['indication'], y_dict['indication'], label = 'indication (AUC=' + '%.3f)' % max_auc['indication'])
#plt.plot(x_dict['MoA'], y_dict['MoA'], label = 'MoA (AUC=' + '%.3f)' % max_auc['MoA'])
#plt.plot(x_dict['sideEffect'], y_dict['sideEffect'], label = 'sideEffect (AUC=' + '%.3f)' % max_auc['sideEffect'])

plt.plot([0, 1], [0, 1], '--', label = 'Random (AUC= 0.50)')
plt.legend()
plt.axis('scaled')
plt.title('DDI')
plt.savefig('/Users/noora/Documents/CBRC2018/Neural_Network/newTest/D4paper/AUCs/GOHPOALL/Twosides-MoA1.png')