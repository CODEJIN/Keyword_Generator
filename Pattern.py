import tensorflow as tf;
import numpy as np;
import pandas as pd;
import _pickle as pickle;
from random import shuffle, choice;
from threading import Thread;
from collections import deque;
import time;
import Hyper_Parameters;

class Feeder:
    def __init__(self, batch_Size, max_Queue= 1000):
        self.batch_Size = batch_Size;
        self.max_Queue = max_Queue;

        self.Load_Pattern();
        self.Placeholder_Generate();        
                
        training_Pattern_Generate_Thread = Thread(target=self.Training_Pattern_Generate);
        training_Pattern_Generate_Thread.daemon = True;
        training_Pattern_Generate_Thread.start();

        test_Pattern_Generate_Thread = Thread(target=self.Test_Pattern_Generate);
        test_Pattern_Generate_Thread.daemon = True;
        test_Pattern_Generate_Thread.start();

    def Placeholder_Generate(self):
        self.placeholder_Dict = {
            'Is_Training': tf.placeholder(tf.bool, name='Is_Training_Placeholder'),
            'Abstract': tf.placeholder(tf.int32, shape=(None, None), name='Abstract_Placeholder'),
            'Keyword': tf.placeholder(tf.int32, shape=(None,), name='Keyword_Placeholder')
            }

    def Load_Pattern(self, file_Path= Hyper_Parameters.Pattern_Path):
        with open(file_Path, 'rb') as f:
            load_Dict = pickle.load(f);

        self.training_Pattern_DataFrame = load_Dict['Train_Dataframe'];
        self.test_Pattern_DataFrame = load_Dict['Test_Dataframe'];
        self.token_Dict = load_Dict['Token_Dict'];

        self.abstract_ID_Size = len(self.token_Dict['Abstract', 'Index', 'Token'])
        self.keyword_ID_Size = len(self.token_Dict['Keyword', 'Index', 'Token'])

        print('Abstract ID size: {}'.format(self.abstract_ID_Size));
        print('Keyword ID size: {}'.format(self.keyword_ID_Size));

    def Training_Pattern_Generate(self):
        self.training_Pattern_Queue = deque();
        
        index_List = [(index, length) for index, length in  enumerate(self.training_Pattern_DataFrame['Abstract.Length'].tolist())]
        if Hyper_Parameters.Pattern_Sorting:
            index_List = sorted(index_List, key= lambda x: x[1])
        index_List = [x for x, _ in index_List]

        while True:
            if not Hyper_Parameters.Pattern_Sorting:
                shuffle(index_List);         
            training_Batch_List = [index_List[start_Index:start_Index+self.batch_Size] for start_Index in range(0, len(index_List), self.batch_Size)];
            shuffle(training_Batch_List)

            current_Index = 0;
            while current_Index < len(training_Batch_List):
                if len(self.training_Pattern_Queue) >= self.max_Queue:
                    time.sleep(0.1);
                    continue;

                batch_Size = len(training_Batch_List[current_Index]);
                batch_Pattern_Dataframe = self.training_Pattern_DataFrame.loc[training_Batch_List[current_Index]];    #batch patterns dataframe

                abstract_Pattern_List = batch_Pattern_Dataframe['Abstract.Pattern'].tolist();
                max_Pattern_Length = max([x.shape[0] for x in abstract_Pattern_List]) + 2
                new_Batch_Abstract_Pattern = np.ones((batch_Size, max_Pattern_Length), dtype=np.int32) * self.token_Dict['Abstract', 'Token', 'Index']['<P>'];
                for index, abstract_Pattern in enumerate(abstract_Pattern_List):
                    new_Batch_Abstract_Pattern[index, 0] = self.token_Dict['Abstract', 'Token', 'Index']['<S>'];
                    new_Batch_Abstract_Pattern[index, 1:abstract_Pattern.shape[0] + 1] = abstract_Pattern;
                    new_Batch_Abstract_Pattern[index, abstract_Pattern.shape[0] + 1] = self.token_Dict['Abstract', 'Token', 'Index']['<E>'];

                new_Batch_Keyword_Pattern = np.array([
                    choice(keyword_Pattern)
                    for keyword_Pattern in batch_Pattern_Dataframe['Keyword.Pattern'].tolist()
                    ], dtype=np.int32)

                new_Feed_Dict = {
                    self.placeholder_Dict['Is_Training']: True,
                    self.placeholder_Dict['Abstract']: new_Batch_Abstract_Pattern,
                    self.placeholder_Dict['Keyword']: new_Batch_Keyword_Pattern
                    }

                self.training_Pattern_Queue.append(new_Feed_Dict);                
                current_Index += 1;

    def Get_Training_Pattern(self):
        while True:
            if len(self.training_Pattern_Queue) > 0:
                break;
            time.sleep(0.1);

        return self.training_Pattern_Queue.popleft();

    def Test_Pattern_Generate(self):
        self.test_Pattern_List = [];

        index_List = list(range(len(self.test_Pattern_DataFrame['Abstract.Length'].tolist())));
        test_Batch_List = [index_List[start_Index:start_Index+self.batch_Size] for start_Index in range(0, len(index_List), self.batch_Size)];

        for test_Batch in test_Batch_List:
            batch_Size = len(test_Batch);
            batch_Pattern_Dataframe = self.test_Pattern_DataFrame.loc[test_Batch];    #batch patterns dataframe
            
            abstract_Pattern_List = batch_Pattern_Dataframe['Abstract.Pattern'].tolist();
            max_Pattern_Length = max([x.shape[0] for x in abstract_Pattern_List]) + 2
            new_Batch_Abstract_Pattern = np.ones((batch_Size, max_Pattern_Length), dtype=np.int32) * self.token_Dict['Abstract', 'Token', 'Index']['<P>'];
            for index, abstract_Pattern in enumerate(abstract_Pattern_List):
                new_Batch_Abstract_Pattern[index, 0] = self.token_Dict['Abstract', 'Token', 'Index']['<S>'];
                new_Batch_Abstract_Pattern[index, 1:abstract_Pattern.shape[0] + 1] = abstract_Pattern;
                new_Batch_Abstract_Pattern[index, abstract_Pattern.shape[0] + 1] = self.token_Dict['Abstract', 'Token', 'Index']['<E>'];

            new_Feed_Dict = {
                self.placeholder_Dict['Is_Training']: False,
                self.placeholder_Dict['Abstract']: new_Batch_Abstract_Pattern
                }

            self.test_Pattern_List.append((batch_Pattern_Dataframe['Keyword.Pattern'].tolist(), new_Feed_Dict));

    def Get_Test_Pattern(self):        
        while not hasattr(self, 'test_Pattern_List'):            
            time.sleep(0.1);
        return self.test_Pattern_List;


    def Get_Test_Pattern_from_Abstract(self, abstract_List):
        max_Abstract_Length = max([len(x.strip().split(' ')) for x in abstract_List]) + 2;  #<S> and <E>

        sparse_Abstract_Pattern_List = [];
        for abstract in abstract_List:
            abstract = [x.strip().rstrip(',').rstrip('.').lstrip('(').rstrip(')') for x in abstract.strip().lower().split(' ')];
            abstract = ['<S>'] + abstract + ['<E>'] + ['<P>'] * (max_Abstract_Length - len(abstract) - 2);   #-2 is about <S> and <E>
            new_Token_Index_List = []
            for token in abstract:
                try:
                    new_Token_Index_List.append(self.token_Dict['Abstract', 'Token', 'Index'][token]);
                except KeyError as e:
                    new_Token_Index_List.append(self.token_Dict['Abstract', 'Token', 'Index']['<UNK>']);
            new_Sparse_Pattern = np.array(new_Token_Index_List, dtype= np.int32);
            sparse_Abstract_Pattern_List.append(new_Sparse_Pattern);

        new_Pattern_Dict = {
            self.placeholder_Dict['Is_Training']: False,
            self.placeholder_Dict['Abstract']: np.stack(sparse_Abstract_Pattern_List)
            }        
        return new_Pattern_Dict

    def Index_to_Keyword(self, index_Array):        
        return [
            self.token_Dict['Keyword', 'Index', 'Token'][index]
            for index in index_Array            
            ]