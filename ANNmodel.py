import pandas as pd
# Import vec file
with open('/home/drnoor/Desktop/AfterHypers/HPO/embedding.lst', 'r') as file:
    text = file.read()
# Strip and split vector data into list of lists [chem, vec]
text = text.replace('\n', '')
text = text.split(']')
text = [item.strip().split(' [') for item in text]
# Turn it into a data frame
df = pd.DataFrame(text)
df.columns = ['ID', 'Vector']
# Clean
df = df.dropna()
df['Vector'] = df.Vector.map(lambda x: x.rstrip().lstrip().replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
# Turn vector column into a list
df['Vector'] = df.Vector.map(lambda x: x.split(','))
for i in range(df['Vector'].shape[0]):
    df['Vector'][i] = pd.to_numeric(df['Vector'][i])
drug_dict = dict(zip(df['ID'][:],df['Vector'][:]))
positives = set()
true_positives = set()
possible_positives = set()
true_pos_drugs = set()

files = ['/home/drnoor/Desktop/AfterHypers/HPO/new-BiologicalProcess.lst']

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
        print(len(positives),len(true_positives), len(possible_positives), len(true_pos_drugs))
    drug_set = set(list(drug_dict))
    print(len(drug_set))
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization, Activation
    from keras.utils import multi_gpu_model, Sequence, np_utils
    import math
    from keras.optimizers import SGD, Adam, RMSprop, Nadam
    from keras.callbacks import EarlyStopping, TensorBoard
    import scipy.stats as ss
    from keras.backend.tensorflow_backend import set_session
    from keras import backend as K
    from hyperopt import Trials, STATUS_OK, tpe
    from hyperas import optim
    from hyperas.distributions import choice, uniform
    import numpy as np
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
    rank_counts = []
    write_counts = []
    epochs = 50
    for i in range(epochs):
        rank_counts.append(dict())
        write_counts.append(dict())
    counter = 0
    for drug in true_pos_drugs:
        K.clear_session()
        counter+=1
        print('drug ', counter)
        val_drugs = set()
        val_drugs.add(drug)
        train_drugs = set(list(drug_set)) - val_drugs
        print('val_drugs: ', val_drugs)
        print('len of train_drugs: ', len(train_drugs))
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
        train_negatives = []
        train_all_tuples = set()
        for drug1 in train_drugs:
            for drug2 in train_drugs:
                if drug1 in drug_dict and drug2 in drug_dict and drug1 != drug2:
                    train_all_tuples.add((drug1, drug2))
        print('len(train_all_tuples):', len(train_all_tuples))
        for item in train_all_tuples:
            if item not in true_positives and (item[1], item[0]) not in true_positives:
                train_negatives.append((item[0], item[1], 0))
        print('len(train_negatives):', len(train_negatives))
        val_negatives = []
        val_all_tuples = set()
        for drug1 in val_drugs:
            for drug2 in train_drugs:
                if drug1 in drug_dict and drug2 in drug_dict:
                    val_all_tuples.add((drug1, drug2))
        print('len(val_all_tuples): ', len(val_all_tuples))
        for item in val_all_tuples:
            if item not in true_positives and (item[1], item[0]) not in true_positives:
                val_negatives.append((item[0], item[1], 0))
        print('len(val_negatives): ', len(val_negatives))
        train_positives = np.repeat(np.array(list(train_positives)), len(train_negatives)//len(train_positives), axis = 0)
        train_negatives = np.array(list(train_negatives))
        triple_train = np.concatenate((train_positives, train_negatives), axis=0)
        np.random.shuffle(triple_train)
        triple_val = np.concatenate((val_positives, val_negatives), axis=0)
        batch_size = 2**12
        num_classes = 1
        model = Sequential()
        model.add(Dense(16, activation='relu', input_shape=(400,)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1825334682554174))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        generator = Generator(triple_train[:len(triple_train),0:2], triple_train[:len(triple_train),2], batch_size)
        val_generator = Generator(triple_val[:,0:2], triple_val[:,2], batch_size)
        for i in range(epochs):    
            history = model.fit_generator(generator=generator,
                                epochs=1,
                                steps_per_epoch = int(math.ceil(len(triple_train)/ batch_size)*0.05),
                                verbose=2,
                                validation_data=val_generator,
                                validation_steps= int(math.ceil(len(triple_val)/ batch_size)))
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
                for other_drug in train_drugs:
                    if other_drug not in positive_set:
                        protein_list.append((drug, other_drug, 0))
                protein_list = np.array(protein_list)
                sim_list = model.predict_generator(generator=Generator(protein_list[:,0:2], protein_list[:,2], 1000), 
                    verbose=2, steps=int(math.ceil(math.ceil(len(protein_list))  / 1000)))
                
                DDI=[]
                for c in range (num_positive, len(protein_list)):
                    DDI.append((protein_list[c][0],protein_list[c][1], sim_list[c][0]))
                output = open(file+'possible', 'a+')
                output.write(str(DDI) + '\n')

                print(sim_list[:num_positive+5])
                y_rank = ss.rankdata(-sim_list, method='average')
                x_list = y_rank[:num_positive]
                print(np.mean(x_list))
                for x in x_list:
                    if x not in rank_counts[i]:
                        rank_counts[i][x] = 0
                    rank_counts[i][x]+=1
                print(np.mean(list(rank_counts[i].keys())))
                write_counts[i][drug] = x_list
    for i in range(epochs):
        auc_x = list(rank_counts[i].keys())
        auc_x.sort()
        auc_y = []
        tpr = 0
        step = 1/sum(rank_counts[i].values())
        for x in auc_x:
            tpr += rank_counts[i][x]*step
            auc_y.append(tpr)
        auc_x.append(len(drug_set)-1)
        auc_y.append(1)
        auc = np.trapz(auc_y, auc_x)/(len(drug_set)-1)
        print('Rank based auc is: %f' % (auc)) 
    
    import pickle

    with open(file+'HPO.pckl', 'wb') as fp:
        pickle.dump((len(drug_set),write_counts), fp)
