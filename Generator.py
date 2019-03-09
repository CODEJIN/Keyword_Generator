import json
import pandas as pd;
import numpy as np;
import _pickle as pickle;
import os, time;
import Hyper_Parameters;
from random import shuffle

abstract_Token_Count_Dict = {};
keyword_Token_Count_Dict = {};
for corpus_Index in range(47):
    st = time.time()    
    file_Name = os.path.join(Hyper_Parameters.Data_Path, 's2-corpus-{:02d}'.format(corpus_Index));
    print(file_Name)
    with open(file_Name, 'r', encoding='utf-8-sig') as f: 
        for line in f.readlines():
            json_Data = json.loads(line)
            for abstract_Token in json_Data['paperAbstract'].strip().split(' '):
                abstract_Token = abstract_Token.strip().lower().rstrip(',').rstrip('.').lstrip('(').rstrip(')');
                if not abstract_Token in abstract_Token_Count_Dict.keys():
                    abstract_Token_Count_Dict[abstract_Token] = 0;
                abstract_Token_Count_Dict[abstract_Token] += 1;
            for keyword_Token in json_Data['entities']:
                keyword_Token = keyword_Token.strip().lower().rstrip(',').rstrip('.').lstrip('(').rstrip(')');
                if not keyword_Token in keyword_Token_Count_Dict.keys():
                    keyword_Token_Count_Dict[keyword_Token] = 0;
                keyword_Token_Count_Dict[keyword_Token] += 1;
    print(time.time() - st)

abstract_Token_List = [key for key, value in abstract_Token_Count_Dict.items() if value >= Hyper_Parameters.Min_Abstract_Token_Entry_Criterion]
keyword_Token_List = [key for key, value in keyword_Token_Count_Dict.items() if value >= Hyper_Parameters.Min_Keyword_Token_Entry_Criterion]

token_Dict = {    
    ('Abstract', 'Index', 'Token'): {index: token for index, token in enumerate(abstract_Token_List, 4)},
    ('Abstract', 'Token', 'Index'): {token: index for index, token in enumerate(abstract_Token_List, 4)},
    ('Keyword', 'Index', 'Token'): {index: token for index, token in enumerate(keyword_Token_List)},
    ('Keyword', 'Token', 'Index'): {token: index for index, token in enumerate(keyword_Token_List)}
    }
token_Dict['Abstract', 'Index', 'Token'].update({0: '<P>', 1: '<S>', 2: '<E>', 3: '<UNK>'});
token_Dict['Abstract', 'Token', 'Index'].update({'<P>': 0, '<S>': 1, '<E>': 2, '<UNK>': 3});

abstract_Pattern_List = [];
keyword_Pattern_List = [];
for corpus_Index in range(47):
    st = time.time()
    file_Name = os.path.join(Hyper_Parameters.Data_Path, 's2-corpus-{:02d}'.format(corpus_Index));
    print(file_Name)
    with open(file_Name, 'r', encoding='utf-8-sig') as f: 
        for line in f.readlines():
            abstract_Index_List = [];
            keyword_Index_List = [];

            json_Data = json.loads(line)
            for abstract_Token in json_Data['paperAbstract'].strip().split(' '):
                abstract_Token = abstract_Token.strip().lower().rstrip(',').rstrip('.').lstrip('(').rstrip(')');
                try:
                    abstract_Index_List.append(token_Dict['Abstract', 'Token', 'Index'][abstract_Token]);
                except KeyError as e:
                    abstract_Index_List.append(token_Dict['Abstract', 'Token', 'Index']['<UNK>']);
            for keyword_Token in json_Data['entities']:
                keyword_Token = keyword_Token.strip().lower().rstrip(',').rstrip('.').lstrip('(').rstrip(')');
                try:
                    keyword_Index_List.append(token_Dict['Keyword', 'Token', 'Index'][keyword_Token]);
                except KeyError as e:
                    pass;
            if len(abstract_Index_List) < Hyper_Parameters.Min_Abstract_Length:
                continue;
            elif len(abstract_Index_List) > Hyper_Parameters.Max_Abstract_Length:
                continue;
            elif len(keyword_Index_List) == 0:
                continue;
            elif len([x for x in abstract_Index_List if x==token_Dict['Abstract', 'Token', 'Index']['<UNK>']])/len(abstract_Index_List) > Hyper_Parameters.Max_UNK_Ratio:
                continue;

            abstract_Pattern_List.append(np.array(abstract_Index_List, dtype=np.int32));
            keyword_Pattern_List.append(np.array(keyword_Index_List, dtype=np.int32));

    print(time.time() - st)

pattern_Count = len(abstract_Pattern_List);
pattern_Index_List = list(range(pattern_Count));
shuffle(pattern_Index_List);
training_Index_List = pattern_Index_List[:int(pattern_Count * (1 - Hyper_Parameters.Test_Ratio))]
test_Index_List = pattern_Index_List[int(pattern_Count * (1 - Hyper_Parameters.Test_Ratio)):]

print('Training pattern count: {}'.format(len(training_Index_List)));
print('Training pattern count: {}'.format(len(test_Index_List)));
print('Abstract token count: {}'.format(len(token_Dict['Abstract', 'Index', 'Token'])));
print('Keyword token count: {}'.format(len(token_Dict['Keyword', 'Index', 'Token'])));


save_Dict = {};
save_Dict['Train_Dataframe'] = pd.DataFrame({    
    "Abstract.Pattern": [abstract_Pattern_List[x] for x in training_Index_List],
    "Keyword.Pattern": [keyword_Pattern_List[x] for x in training_Index_List],
    "Abstract.Length": [len(abstract_Pattern_List[x]) for x in training_Index_List]
    }) 
save_Dict['Test_Dataframe'] = pd.DataFrame({
    "Abstract.Pattern": [abstract_Pattern_List[x] for x in test_Index_List],
    "Keyword.Pattern": [keyword_Pattern_List[x] for x in test_Index_List],
    "Abstract.Length": [len(abstract_Pattern_List[x]) for x in test_Index_List]
    }) 
save_Dict['Token_Dict'] = token_Dict;

with open(Hyper_Parameters.Pattern_Path, 'wb') as f:
    pickle.dump(save_Dict, f, protocol=2);