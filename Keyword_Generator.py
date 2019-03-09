import os, time;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np;
import tensorflow as tf;
import _pickle as pickle;
import matplotlib.pyplot as plt;
import Hyper_Parameters;
import Pattern;
from Module import Embedding, BiLSTM, Multihead_Attention, Index_Embedding, Sinusoid_Position_Emebedding;

class Keyword_Generator:
    def __init__(self):
        self.tf_Session = tf.Session();
        self.feeder = Pattern.Feeder(batch_Size= Hyper_Parameters.Batch_Size);
        self.Tensor_Generate();
        self.tf_Saver = tf.train.Saver(max_to_keep=5);

    def Tensor_Generate(self):
        placeholder_Dict = self.feeder.placeholder_Dict;

        self.placeholder_Dict = {
            'Is_Training': tf.placeholder(tf.bool, name='Is_Training_Placeholder'),
            'Abstract': tf.placeholder(tf.int32, shape=(None, None), name='Abstract_Placeholder'),
            'Keyword': tf.placeholder(tf.int32, shape=(None,), name='Keyword_Placeholder')
            }

        #Index and position embedding
        with tf.variable_scope('model') as scope:
            encoder_Embedding_Tensor = Embedding(
                inputs= placeholder_Dict['Abstract'],
                id_size= self.feeder.abstract_ID_Size,
                max_Time= Hyper_Parameters.Max_Abstract_Length + 2, #<S> and <E>
                embedding_size = Hyper_Parameters.Embedding_Size,
                trainable= True
                )

            #Dropout
            encoder_Embedding_Tensor = tf.layers.dropout(
                encoder_Embedding_Tensor,
                rate= Hyper_Parameters.Dropout_Rate,
                training= placeholder_Dict['Is_Training'],
                name='dropout'
                )

            encoder_Tensor = BiLSTM(
                inputs= encoder_Embedding_Tensor,
                input_length= None,
                cell_Size= Hyper_Parameters.LSTM_Size,
                name= 'biLSTM'
                )
            
            encoder_Tensor, attention_Visualization_Tensor = Multihead_Attention(
                queries= tf.reduce_mean(encoder_Tensor, axis=1, keepdims=True),
                keys= encoder_Tensor,
                attention_size= Hyper_Parameters.Attention_Size,
                head_size= Hyper_Parameters.Head_Size,
                future_masking= False,
                dropout_Rate= Hyper_Parameters.Dropout_Rate,
                is_Training= placeholder_Dict['Is_Training'],
                name= 'multihead_Attention'
                )

            #Linear projection
            logits = tf.layers.dense(
                tf.squeeze(encoder_Tensor, axis=1),
                self.feeder.keyword_ID_Size,
                name='logits'
                )

        with tf.variable_scope('loss') as scope:
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels= tf.one_hot(placeholder_Dict['Keyword'], self.feeder.keyword_ID_Size),
                logits= logits,
                label_smoothing= 0.1
                )
            
            global_Step = tf.Variable(0, name='global_step', trainable = False);

            #Noam decay
            step = tf.cast(global_Step + 1, dtype=tf.float32);
            warmup_Steps = 4000.0;
            learning_Rate = (Hyper_Parameters.Attention_Size * Hyper_Parameters.Head_Size) ** -0.5 * \
                tf.minimum(step * warmup_Steps**-1.5, step**-0.5)

            #Adam
            optimizer = tf.train.AdamOptimizer(learning_Rate, beta1= 0.9, beta2= 0.98, epsilon= 1e-09);
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 1.0)
            optimize = optimizer.apply_gradients(zip(clipped_Gradients, variables), global_step=global_Step)

        with tf.variable_scope('test') as scope:
            predictions = tf.math.top_k(logits, k= Hyper_Parameters.Num_Extract_Keyword).indices

        self.training_Tensor_Dict = {
            'Global_Step': global_Step,
            'Learning_Rate': learning_Rate,
            'Loss': loss,
            'optimize': optimize
            }
        self.test_Tensor_Dict = {
            'Global_Step': global_Step,
            'Predictions': predictions,
            'Attentions': tf.squeeze(attention_Visualization_Tensor, axis=-1)
            }

        self.tf_Session.run(tf.global_variables_initializer());

    def Restore(self):        
        checkpoint_Path = os.path.join(Hyper_Parameters.Save_Path, 'Checkpoint').replace('\\', '/');
        os.makedirs(checkpoint_Path, exist_ok= True);

        checkpoint_Path = tf.train.latest_checkpoint(checkpoint_Path);
        print('Lastest checkpoint:', checkpoint_Path);

        if checkpoint_Path is None:
            print('There is no checkpoint');
        else:
            self.tf_Saver.restore(self.tf_Session, checkpoint_Path);
            print('Checkpoint \'', checkpoint_Path, '\' is loaded');

    def Train(self, test_Timing= Hyper_Parameters.Test_Timing):
        self.External_Test();
        self.Test();
        while True:
            start_Time = time.time();
            result_Dict = self.tf_Session.run(
                fetches= self.training_Tensor_Dict,
                feed_dict= self.feeder.Get_Training_Pattern()
                )
            print('Spent_Time: {:.5f}\t\tGlobal step: {}\t\tLearning rate: {:.5f}\t\tLoss: {:.5f}'.format(
                time.time() - start_Time,
                result_Dict['Global_Step'],
                result_Dict['Learning_Rate'],
                result_Dict['Loss']
                ))

            if test_Timing > 0 and result_Dict['Global_Step'] % test_Timing == 0:
                self.Checkpoint_Save(result_Dict['Global_Step']);
                self.External_Test();
                self.Test();

    def Checkpoint_Save(self, global_Step):
        checkpoint_Path = os.path.join(Hyper_Parameters.Save_Path, 'Checkpoint').replace('\\', '/');
        os.makedirs(checkpoint_Path, exist_ok= True);
        self.tf_Saver.save(self.tf_Session, os.path.join(checkpoint_Path, 'Checkpoint').replace('\\', '/'), global_step=global_Step);
        print('Checkpoint saved');

    def Test(self):
        extract_List = ['Num_of_Correct\tNum_of_Target\tAccuracy\tTarget\tPrediction'];

        accuracy_List = [];

        test_Pattern_List = self.feeder.Get_Test_Pattern();
        for target_Keyword_List, feed_Dict in test_Pattern_List:
            result_Dict = self.tf_Session.run(
                fetches= self.test_Tensor_Dict,
                feed_dict= feed_Dict
                )

            for target, prediction in zip(target_Keyword_List, result_Dict['Predictions']):
                extract_List.append('{}\t{}\t{}\t{}\t{}'.format(
                    len(set(target) & set(prediction)),
                    len(target),
                    len(set(target) & set(prediction)) / min(len(target), Hyper_Parameters.Num_Extract_Keyword),
                    self.feeder.Index_to_Keyword(target),
                    self.feeder.Index_to_Keyword(prediction)
                    ).replace('[', '').replace(']', ''))
                accuracy_List.append(len(set(target) & set(prediction)) / min(len(target), Hyper_Parameters.Num_Extract_Keyword))

        test_Path = os.path.join(Hyper_Parameters.Save_Path, 'Test').replace('\\', '/');
        os.makedirs(test_Path, exist_ok= True);
        with open(os.path.join(test_Path, '{}.txt'.format(str(result_Dict['Global_Step']))).replace('\\', '/'), 'w', encoding='utf-8-sig') as f:
            f.write('\n'.join(extract_List));

        with open(os.path.join(test_Path, 'Accuracy.txt').replace('\\', '/'), 'a', encoding='utf-8-sig') as f:
            f.write('{}\t{}\n'.format(result_Dict['Global_Step'], np.mean(accuracy_List)))

    def External_Test(self):
        abstract_List = [];
        with open('External_Test.txt', 'r') as f:
            for line in f.readlines():
                abstract_List.append(line);

        result_Dict = self.tf_Session.run(
            fetches= self.test_Tensor_Dict,
            feed_dict= self.feeder.Get_Test_Pattern_from_Abstract(abstract_List)
            )

        test_Path = os.path.join(Hyper_Parameters.Save_Path, 'External_Test').replace('\\', '/');
        os.makedirs(test_Path, exist_ok= True);

        for index, (abstract, attention_Array, predictions) in enumerate(zip(abstract_List, result_Dict['Attentions'], result_Dict['Predictions'])):
            abstract = ['<S>'] + [x.strip().rstrip(',').rstrip('.') for x in abstract.strip().lower().split(' ')] + ['<E>'];
            fig = plt.figure(figsize=(attention_Array.shape[1] / 2, attention_Array.shape[0] / 2));
            plt.imshow(attention_Array, cmap='Blues', aspect='auto', interpolation='none');
            plt.xlabel("Text");
            plt.ylabel("Head");
            plt.gca().set_xticks(range(len(abstract)));
            plt.gca().set_xticklabels([x for x in abstract], rotation=40, ha='right');
            plt.title("Attention focus    Step: {}            Extracted keywords: {}".format(result_Dict['Global_Step'], ', '.join(self.feeder.Index_to_Keyword(predictions))))
            plt.savefig(
                os.path.join(test_Path, '{}.Idx_{}.Attention.png'.format(result_Dict['Global_Step'], index)).replace('\\', '/'),
                bbox_inches='tight'
                )
            
        with open(os.path.join(test_Path, '{}.txt'.format(str(result_Dict['Global_Step']))).replace('\\', '/'), 'w', encoding='utf-8-sig') as f:
            f.write('n'.join(['\t'.join(self.feeder.Index_to_Keyword(predictions)) for predictions in result_Dict['Predictions']]))

if __name__ == '__main__':
    new_Keyword_Generator = Keyword_Generator();
    new_Keyword_Generator.Restore();
    new_Keyword_Generator.Train();