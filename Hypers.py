from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import Sequence
import math
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras import backend as K
import scipy.stats as ss


def data():
    
    with open('/Users/adeebnoor/Documents/CBRC2018/Neural_Network/newTest/GOHPOALL/embedding.lst', 'r') as file:
        text = file.read()
        text = text.replace('\n', '')
        text = text.split(']')
        text = [item.strip().split(' [') for item in text]
        df = pd.DataFrame(text)
        df.columns = ['ID', 'Vector']
        df = df.dropna()
        df['Vector'] = df.Vector.map(lambda x: x.rstrip().lstrip().replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        df['Vector'] = df.Vector.map(lambda x: x.split(','))
        for i in range(df['Vector'].shape[0]):
            df['Vector'][i] = pd.to_numeric(df['Vector'][i])
        drug_dict = dict(zip(df['ID'][:], df['Vector'][:]))

    positives = set()  # drug found with embedding 
    true_positives = set()  #  found with embedding and DDI
    possible_positives = set()  # found embedding but not DDI 
    true_pos_drugs = set()  # list of drugs found with embedding and DDIS 

    files = ['/Users/adeebnoor/Documents/CBRC2018/Neural_Network/newTest/MoA/new-transporterInhibtor.lst']

    for file in files:    
        with open(file, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                if items[0] in drug_dict and items[1] in drug_dict:
                    positives.add((items[0], items[1]))
                    if items[402] == '0':
                        possible_positives.add((items[0], items[1]))
                    else:
                        true_pos_drugs.add(items[0])
                        true_pos_drugs.add(items[1])
                        true_positives.add((items[0], items[1]))
            print ('explained by rule = ', len(positives))
            print ('explained by rule and true DDIs = ', len(true_positives))
            print ('explained by rule but not DDIs = ', len(possible_positives))
            print ('num of drugs explained by rule and true DDIs = ', len(true_pos_drugs))

        drug_set = set(list(drug_dict))
        print('embedding size = ', len(drug_set))
    
    return drug_set, positives, true_positives, possible_positives, true_pos_drugs



def create_model(drug_dict, true_positives, true_pos_drugs, drug_set):

    class Generator(Sequence):
        
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
            self.nbatch = int(np.ceil(len(self.x) / float(self.batch_size)))
            self.length = len(self.x)
        
        def __len__(self):
            return self.nbatch
        
        def __getitem__(self, idx):
            start = idx * self.batch_size
            batch_len = min(self.batch_size, (self.length)-start)
            X_batch_list = np.empty((batch_len, 400), dtype=np.float32)
            y_batch_list = np.empty(batch_len, dtype=np.float32)
            for ids in range(start, min((idx + 1) * self.batch_size, self.length)):
                array1 = drug_dict[self.x[ids][0]]
                array2 = drug_dict[self.x[ids][1]]
                embds = np.concatenate([array1, array2])
                X_batch_list[ids-start] = embds
                y_batch_list[ids-start] = self.y[ids]
            return X_batch_list, y_batch_list
        
    
    batch_size = 2**11
    rank_counts = []
    epochs = 100
    for i in range(epochs):
        rank_counts.append(dict()) 
        
    model = Sequential()
    model.add(Dense(units={{choice([256, 128, 64, 32, 16])}}, activation={{choice(['relu', 'sigmoid'])}}, input_shape=(400,)))
    model.add(BatchNormalization())
    model.add(Dropout(rate={{uniform(0, 1)}}))
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(units={{choice([32, 16, 8, 4])}}, activation={{choice(['relu', 'sigmoid'])}}))
        model.add(BatchNormalization())
        model.add(Dropout(rate = {{uniform(0, 1)}}))
        if {{choice(['three', 'four'])}} == 'three':
            model.add(Dense(units={{choice([8, 4, 2])}}, activation={{choice(['relu', 'sigmoid'])}}))
            model.add(BatchNormalization())
            model.add(Dropout(rate={{uniform(0, 1)}}))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam'])}},
                  metrics=['accuracy'])
    
    model.save('/Users/adeebnoor/Documents/CBRC2018/Neural_Network/newTest/hypers/model_200.h5')
    counter = 0
    for drug in true_pos_drugs:
        K.clear_session()
        model = load_model('/Users/adeebnoor/Documents/CBRC2018/Neural_Network/newTest/hypers/model_200.h5')
        counter+=1
        print()
        print('drug ', counter)
        val_drugs = set()
        val_drugs.add(drug)
        train_drugs = set(list(drug_set)) - val_drugs
        print('val_drugs: ', val_drugs)
        print('number of left drug in embedding: ', len(train_drugs))
        
        'create positives (training and validation from hard proven (DDI + MoA) = true_positives'
        'train positives = DDIs but not include the val drug'
        'val positives = DDIs and include the val drug'
        train_positives = []
        val_positives = []
        train_positives_set = set()
        val_positives_set = set()
        for items in true_positives:
            if items[1] not in val_drugs and items[0] not in val_drugs:
                train_positives_set.add((items[0], items[1]))
                train_positives.append((items[0], items[1], 1))
            if items[1] in val_drugs or items[0] in val_drugs:
                val_positives_set.add((items[0], items[1]))
                val_positives.append((items[0], items[1], 1))
        print('len(train_positives), len(val_positives): ', len(train_positives), len(val_positives))
        
        'create negatives from embedding- random DDIs'
        train_negatives = []
        train_all_tuples = set()
        for drug1 in train_drugs:
            for drug2 in train_drugs:
                if drug1 in drug_dict and drug2 in drug_dict and drug1 != drug2:
                    train_all_tuples.add((drug1, drug2))
        print('len(train_all_tuples):', len(train_all_tuples))
       
        'filter random DDIs from gold-standard to create train-negatives'
        for item in train_all_tuples:
            if item not in true_positives and (item[1], item[0]) not in true_positives:
                train_negatives.append((item[0], item[1], 0))
        print('len(train_negatives (negative DDIs):', len(train_negatives))
        
        train_positives = np.repeat(np.array(list(train_positives)), len(train_negatives)//len(train_positives), axis = 0)
        train_negatives = np.array(list(train_negatives))
        triple_train = np.concatenate((train_positives, train_negatives), axis=0)
        np.random.shuffle(triple_train)
        
        factor = 1
        generator = Generator(triple_train[:int(factor*len(triple_train)),0:2], triple_train[:int(factor*len(triple_train)),2], batch_size)
                
        
        for i in range(epochs):
            history = model.fit_generator(generator=generator,
                                epochs=100,
                                steps_per_epoch = int(math.ceil(math.ceil(factor*len(triple_train))/ batch_size)),
                                verbose=1,
                                validation_data=generator,
                                validation_steps=1)

            for drug in val_drugs:
                protein_list = []
                positive_set = set()
                for items in true_positives:
                    if items[1] == drug:
                        protein_list.append((items[0], items[1], 1))
                        positive_set.add(items[0])
                    elif items[0] == drug:
                        protein_list.append((items[0], items[1], 1))
                        positive_set.add(items[1])
                num_positive = len(protein_list)
                
                DDI = []

                for other_drug in train_drugs:
                    if other_drug not in positive_set:
                        protein_list.append((drug, other_drug, 0))
                protein_list = np.array(protein_list) # name of the drug
                sim_list = model.predict_generator(generator=Generator(protein_list[:,0:2], protein_list[:,2], 1000),
                                                    verbose=2, steps=int(math.ceil(math.ceil(len(protein_list))  / 1000)))
                
                y_rank = ss.rankdata(-sim_list, method='average')
                x_list = y_rank[:num_positive]
                print(np.mean(x_list))
                for x in x_list:
                    if x not in rank_counts[i]:
                        rank_counts[i][x] = 0
                    rank_counts[i][x]+=1
                
                for i in range(num_positive,len(protein_list)):
                    DDI.append((protein_list[i][0],protein_list[i][1],sim_list[i][0]))
            
            output = open('/Users/adeebnoor/Documents/CBRC2018/Neural_Network/newTest/hypers/possible.txt', 'a+')
            output.write(str(DDI) + '\n')            
        
    
    aucs = []
    for i in range(epochs):                
        auc_x = list(rank_counts[i].keys())
        auc_x.sort()
        auc_y = []
        tpr = 0
        step = 1 / sum(rank_counts[i].values())
        for x in auc_x:
            tpr += rank_counts[i][x] * step
            auc_y.append(tpr)
        auc_x.append(len(drug_set))
        auc_y.append(1)
        auc1 = np.trapz(auc_y, auc_x) / len(drug_set)
        print('Rank based auc is: %f' % (auc1))
        aucs.append(auc1)
    max_auc = max(aucs)
    output = open('/Users/adeebnoor/Documents/CBRC2018/Neural_Network/newTest/hypers/hyperopt_200.aucs', 'a+')
    output.write(str(aucs) + '\n')
    return {'loss':-max_auc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=50, trials=Trials())
    print("Evaluation of best performing model:")
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
