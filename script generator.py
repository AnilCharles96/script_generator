# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:29:34 2019

@author: Anil
"""
# libraries
import urllib
from collections import Counter
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.contrib import seq2seq
import time
import sys

# sherlock holmes txt file 
url = 'https://www.gutenberg.org/files/48320/48320-0.txt'
text = urllib.request.urlopen(url).read().decode('utf-8')

# removing unwanted contents
text = text[3215:]
text = str(text)

    
# convert punctuation to corresponding words  
def token_lookup():
    
    dict = {}
    dict['.'] = '||period||'
    dict[','] = '||comma||'
    dict['”'] = '||quotationmark||'
    dict['“'] = '||quotationmark||'
    dict['"'] = '||quotationmark||'
    dict[':'] = '||colon||'
    dict['\''] = '||singlequote||'
    dict[';'] = '||semicolon||'
    dict['!'] = '||exclamationmark||'
    dict['?'] = '||questionmark||'
    dict['('] = '||leftparentheses||'
    dict[')'] = '||rightparentheses||'
    dict['--'] = '||dash||'
    dict['_'] = '||underscore||'
    dict['—'] = '||hyphen||'
    dict['\n'] = '||return||'
    dict['$'] = '||dollar||'
    dict['['] = '||bracket||'
    dict[']'] = '||bracket||'
    dict['/'] = '||slash||'
    dict['&'] = '||ampersand||'
    dict['*'] = '||star||'
    dict['‘'] = '||singlequote||'
    dict['’'] = '||singlequote||'
    

    return dict

# replacing punctuation to words in text
token_dict = token_lookup()
for key,value in token_dict.items():
    text = text.replace(key,' {} '.format(value))

# lowercase and spliting by words
text_cleaned = text.lower().split()

# words to int and int to words
word_counts = Counter(text_cleaned)
sorted_vocab = sorted(word_counts,reverse=True)
int_to_vocab = {i:word for i,word in enumerate(sorted_vocab)}
vocab_to_int = {word:i for i,word in int_to_vocab.items()}
int_text = [vocab_to_int[word] for word in text_cleaned]

# saving cleaned text
pickle.dump((int_to_vocab,vocab_to_int,int_text),open('sherlock_cleaned','wb'))

# loding cleaned text
int_to_vocab,vocab_to_int,int_text = pickle.load(open('sherlock_cleaned','rb'))

vocab_size = len(int_to_vocab)

# hyperparameters
num_epoch = 300
batch_size = 256
rnn_size = 1024
seq_length = 16
lr = 0.001

# checking for cpu/gpu
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

graph = tf.Graph()
with graph.as_default():
    inputs = tf.placeholder(tf.int32,[None,None],name='inputs')
    targets = tf.placeholder(tf.int32,[None,None],name='target')
    #learning_rate = tf.placeholder(tf.float32,name='learning_rate')

    input_shape = tf.shape(inputs)
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=1.0)
    cell = tf.contrib.rnn.MultiRNNCell([drop])
    initial_state = cell.zero_state(input_shape[0],tf.float32)
    initial_state = tf.identity(initial_state,name = 'initial_state')
    
    embedding = tf.Variable(tf.random_normal((vocab_size,rnn_size),-1,1))
    embed = tf.nn.embedding_lookup(embedding,inputs)

    outputs,final_state = tf.nn.dynamic_rnn(cell,embed,dtype=tf.float32)
    final_state = tf.identity(final_state,name='final_state')
    logits = tf.contrib.layers.fully_connected(outputs,vocab_size,activation_fn=None)
    
    probs = tf.nn.softmax(logits,name='probs')


    input_shape = tf.shape(inputs)
    cost = seq2seq.sequence_loss(logits,targets,tf.ones([input_shape[0],input_shape[1]]))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    
    gradients = optimizer.compute_gradients(cost)
    gradient_clipping = [(tf.clip_by_value(grad,-1.,1.),var) for grad,var in gradients]
    train_op = optimizer.apply_gradients(gradient_clipping)



def batches(int_text,batch_size,seq_length):
    
    n_batches = int(len(int_text) / (batch_size * seq_length))
    
    x_data = np.array([int_text[:n_batches * batch_size * seq_length]])
    y_data = np.array([int_text[1:n_batches * batch_size * seq_length + 1]])
    
    x_batches = np.split(x_data.reshape(batch_size,-1),n_batches,1)
    y_batches = np.split(y_data.reshape(batch_size,-1),n_batches,1)
    
    return np.array(list(zip(x_batches,y_batches)))
    
    
batches = batches(int_text,batch_size,seq_length)

    

with tf.Session(graph=graph) as sess:
    
    saver = tf.train.Saver()
    try :
        load = saver.restore(sess,tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
    except:
        sess.run(tf.global_variables_initializer())

    for epoch in range(num_epoch):
        
        state = sess.run(initial_state,feed_dict={inputs:batches[0][0]})
        
        for batch_i,(x_batch,y_batch) in enumerate(batches):
            
            feed_dict = {inputs:x_batch,targets:y_batch,initial_state:state}
            train_loss,state,_ = sess.run([cost,final_state,train_op],feed_dict=feed_dict)
            
            if (epoch * len(batches) + batch_i) % 11 == 0:
                print('Epoch {} batch {}/{} train_loss = {:.3f}'.format(epoch,batch_i,len(batches),train_loss))

    
    saver.save(sess,'model.ckpt')
    print('model trained and saved')
    
    
def replace_tokens():
    
    dict = {}
    dict['.'] = '||period||'
    dict[','] = '||comma||'
    dict['"'] = '||quotationmark||'
    dict[':'] = '||colon||'
    dict['\''] = '||singlequote||'
    dict[';'] = '||semicolon||'
    dict['!'] = '||exclamationmark||'
    dict['?'] = '||questionmark||'
    dict['('] = '||leftparentheses||'
    dict[')'] = '||rightparentheses||'
    dict['--'] = '||dash||'
    dict['_'] = '||underscore||'
    dict['—'] = '||hyphen||'
    dict['\n'] = '||return||'
    dict['$'] = '||dollar||'
    dict['['] = '||leftbracket||'
    dict[']'] = '||rightbracket||'
    dict['/'] = '||slash||'
    dict['&'] = '||ampersand||'
    dict['*'] = '||star||'

    return dict


def replace_space():
    
    dict ={}
    dict['" '] = '"'
    dict[' "'] = '"'
    dict[' ,'] = ','
    dict[', '] = ', '
    dict[' .'] = '.'
    dict[' \''] = '\''
    dict['\' '] = '\''
    dict[' ('] = '('
    dict['( '] = '('
    dict[' )'] = ')'
    dict[')'] = ')'
    dict[' :']  = ':'
    dict[': ']  = ':'
    dict[' —'] = '—'
    dict['— '] = '—'
    dict[' ;'] = ';'
    dict['; '] = ';'
    dict[' &'] = '&'
    dict['& '] = '&'
    dict[' ['] = '['
    dict['[ '] = '['
    dict[' ]'] = ']'
    dict['] '] = ']'

    
    return dict
    
    

script_length = 200
word = 'danger'

new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    
    inputs_tensor = new_graph.get_tensor_by_name('inputs:0')
    initial_state_tensor = new_graph.get_tensor_by_name('initial_state:0')
    final_state_tensor = new_graph.get_tensor_by_name('final_state:0')
    probs_tensor = new_graph.get_tensor_by_name('probs:0')
    
    prev_state = sess.run(initial_state_tensor,feed_dict={inputs_tensor:np.array([[1]])})
    
    generate_word = [word]
    for i in range(script_length):
        
        word_input = [[vocab_to_int[word] for word in generate_word[-seq_length:]]]
        
        probability, prev_state = sess.run([probs_tensor,final_state_tensor],feed_dict={inputs_tensor:word_input,initial_state_tensor:prev_state})       
        a =  probability[0]
 
        pred_word = int_to_vocab[int(np.argmax(a[len(word_input[0])-1]))]
        
        generate_word.append(pred_word)
    
    replace_token_dict = replace_tokens()
    remove_space = replace_space()
    script = ' '.join(generate_word)
    for key,value in replace_token_dict.items():
        #end = '' if key in ['"']
        script = script.replace(value,key)
    for key,value in remove_space.items():
        script = script.replace(key,value)
        
    
    
    time.sleep(10)
    print('\n')
    for word in script:
        sys.stdout.write(word)
        sys.stdout.flush()
        time.sleep(0.01)




        
        
        
        