Data_Path = 'E:/ORC_Data/'  #Set the unzipped ORC data path.
Min_Abstract_Token_Entry_Criterion = 5000   #In all abstracts, all words that appear less than this number will be 'UNK'.
Min_Keyword_Token_Entry_Criterion = 5000    ##In all papers, all keywords that appear less than this number will not be used.
Min_Abstract_Length = 100
Max_Abstract_Length = 400
Max_UNK_Ratio = 0.05    #The 'UNK' ratio is over this value in a sentence, the sentence will not be used.
Test_Ratio = 0.001
Pattern_Path = 'E:/ORC_Data/Pattern.ORC.pickle'

Save_Path = 'E:/Keyword_Generator_Result';
Batch_Size = 64
Test_Timing = 5000
Embedding_Size = 128
LSTM_Size = 256     #LSTM_Size * 2 == Attention * Head_Size
Dropout_Rate = 0.1
Attention_Size = 128    #LSTM_Size * 2 == Attention * Head_Size
Head_Size = 4   #LSTM_Size * 2 == Attention * Head_Size
Num_Extract_Keyword  = 10   #For top K
Pattern_Sorting = True  #When it is True, all patterns within a single mini batch are as similar length as possible, resulting in increased learning speed.
                        #When not in use, every pattern in every epoch is completely random in order.

