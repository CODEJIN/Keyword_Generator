#Reference: https://github.com/Kyubyong/transformer/blob/master/modules.py

import tensorflow as tf;
import numpy as np;
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, OutputProjectionWrapper;
from tensorflow.contrib.seq2seq import AttentionWrapper, BasicDecoder, dynamic_decode, Helper, TrainingHelper;

def Index_Embedding(inputs, id_size, embedding_size, trainable= False, first_row_zero= False, scale= False, name= None, reuse= None):
    with tf.variable_scope(name or 'embedding', reuse=reuse):
        embedding_V = tf.get_variable(
            name = 'embedding_V',
            shape = (id_size, embedding_size),
            dtype = tf.float32,
            trainable = trainable
            )

        if first_row_zero:
            embedding_V = tf.concat([tf.zeros(shape=(1, embedding_size)), embedding_V[1:]], axis=0)

        lookup = tf.nn.embedding_lookup(
            params= embedding_V,
            ids= inputs
            )

        if scale:
            lookup = lookup * (embedding_size ** 0.5)

    return lookup;

def Embedding(inputs, id_size, embedding_size, max_Time, trainable= False, name= None, reuse= None):
    with tf.variable_scope(name or 'embedding', reuse= reuse) as scope:
        #Embedding
        embedded_Index = Index_Embedding(
            inputs= inputs,
            id_size= id_size,
            embedding_size = embedding_size,
            trainable= trainable,
            first_row_zero= True,
            scale= True,
            name= 'Embedded_Index'
            )

    return embedded_Index;
    
def Multihead_Attention(queries, keys, attention_size, head_size, future_masking= False, dropout_Rate= 0.5, is_Training=False, name= None, reuse= None):
    with tf.variable_scope(name or 'multihead_attention', reuse= reuse):
        if queries.get_shape()[-1] != attention_size * head_size:
            raise ValueError('\'attention_size({}) * head_size({})\' must be same the chennel size of \'queries\' tensor({}).'.format(attention_size, head_size, queries.get_shape()[-1]))

        #Projection
        query_Tensor = tf.layers.dense(queries, attention_size * head_size, activation= tf.nn.relu);    #[Batch, Query_Time, Attention_Size * Head]
        key_Tensor, value_Tensor = tf.split(tf.layers.dense(keys, attention_size * head_size * 2, activation= tf.nn.relu), 2, axis=-1); #[Batch, Key_Time, Attention_Size * Head], [Batch, Key_Time, Attention_Size * Head]
         
        #Reshape with head
        reshaped_Query_Tensor = tf.concat(tf.split(query_Tensor, head_size, axis=-1), axis= 0); #[Batch * Head, Query_Time, Attention_Size]
        reshaped_Key_Tensor = tf.concat(tf.split(key_Tensor, head_size, axis=-1), axis= 0);  #[Batch * Head, Key_Time, Attention_Size]
        reshaped_Value_Tensor = tf.concat(tf.split(value_Tensor, head_size, axis=-1), axis= 0);  #[Batch * Head, Key_Time, Attention_Size]

        #Matmul
        output_Tensor = tf.matmul(reshaped_Query_Tensor, tf.transpose(reshaped_Key_Tensor, (0, 2, 1)))  #[Batch * Head, Query_Time, Key_Time]

        #Scale
        output_Tensor /= attention_size ** 0.5

        #Key masking
        #-inf or 1 mask. If key is all zero, the mask is -inf. If not, it is 1.
        #I think this is to avoid to focus to first row.... 
        key_Mask = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))    #[Batch, Key_Time]
        key_Mask = tf.tile(key_Mask, [head_size, 1])   #[Batch * Head, Key_Time]
        key_Mask = tf.tile(tf.expand_dims(key_Mask, 1), [1, tf.shape(queries)[1], 1])   #(Batch * Head, Query_Time, Key_Time)        
        padding_Tensor = tf.ones_like(output_Tensor) * -np.inf
        output_Tensor = tf.where(tf.equal(key_Mask, 0), padding_Tensor, output_Tensor)   #(Batch * Head, Query_Time, Key_Time)

        #Future masking
        if future_masking:
            future_Mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(output_Tensor[0])).to_dense();
            future_Mask = tf.tile(tf.expand_dims(future_Mask, 0), [tf.shape(output_Tensor)[0], 1, 1]);
            padding_Tensor = tf.ones_like(output_Tensor) * -np.inf
            output_Tensor = tf.where(tf.equal(future_Mask, 0), padding_Tensor, output_Tensor)   #(Batch * Head, Query_Time, Key_Time)

        #Softmax
        output_Tensor = tf.nn.softmax(output_Tensor)     #(Batch * Head, Query_Time, Key_Time)        
        
        visulization_Tensor = tf.transpose(tf.stack(tf.split(output_Tensor, head_size, axis=0), axis=1), [0, 1, 3, 2]);    #(Batch, Head, Key_Time, Query_Time)        

        #Query masking
        query_Mask = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))    #[Batch, Query_Time]
        query_Mask = tf.tile(query_Mask, [head_size, 1])   #[Batch * Head, Query_Time]
        query_Mask = tf.tile(tf.expand_dims(query_Mask, -1), [1, 1, tf.shape(keys)[1]])   #(Batch * Head, Query_Time, Key_Time)
        output_Tensor *= query_Mask   #(Batch * Head, Query_Time, Key_Time)

        # Dropouts
        output_Tensor = tf.layers.dropout(output_Tensor, rate= dropout_Rate, training= is_Training);

        #Weighted sum
        output_Tensor = tf.matmul(output_Tensor, reshaped_Value_Tensor)     #(Batch * Head, Query_Time, Attention_Size)

        #Reshape
        output_Tensor = tf.concat(tf.split(output_Tensor, head_size, axis=0), axis=-1);     #(Batch, Query_Time, Attention_Size * Head)

        #Residual
        output_Tensor += queries;
        
        return normalize(output_Tensor, reuse= reuse), visulization_Tensor;

def normalize(inputs, name= None, reuse= None):
    '''
    inputs: [Batch, Time, Size]
    '''
    with tf.variable_scope(name or 'normalize', reuse= reuse):        
        mean_Tensor, variance_Tensor = tf.nn.moments(inputs, axes=[-1], keep_dims=True)    #[Batch, Time]
        normalized_Tensor = (inputs - mean_Tensor) / tf.sqrt(variance_Tensor + 1e-8)  #[Batch, Time, Size]

        size = inputs.get_shape()[-1]
        beta= tf.Variable(tf.zeros((size)), name= 'beta')
        gamma = tf.Variable(tf.ones((size)), name= 'gamma')

        return gamma * normalized_Tensor + beta;
    
def Label_Smoothing(labels, epsilon=0.1):
    k = labels.get_shape().as_list()[-1]
    return ((1-epsilon) * labels) + (epsilon / k)
    
def BiLSTM(inputs, input_length, cell_Size, is_training= False, name= None, reuse= None):
    with tf.variable_scope(name or 'biLSTM', reuse= reuse):
        output_Pattern_List, rnn_State_List = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = LSTMCell(cell_Size),
            cell_bw = LSTMCell(cell_Size),
            inputs = inputs,
            sequence_length = input_length,
            dtype = tf.float32,
            scope = "biLSTM"
        )

        speaker_Embedded_BiLSTM_Activation = tf.concat(
            list(output_Pattern_List),
            axis= -1
            )

    return speaker_Embedded_BiLSTM_Activation;