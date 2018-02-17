import tensorflow as tf
import numpy as np
from optparse import OptionParser
import nltk
nltk.download('punkt')

#set hyperparameters
max_len = 40
step = 2
num_units = 384
learning_rate = 0.001
batch_size = 200
epoch = 120
temperature = 0.5

def read_data(file_name):
    '''
     open and read text file
    '''
    text = open(file_name, 'r').read()
    out = ''
    for char in text:
        if(ord(char)<128):
            out += char
    return out.lower()


def featurize(text):
    '''
     featurize the text to train and target dataset
    '''
    pattern = '(\d+[a-z]+|[a-z]+|\s|\.|,|:|;|-)'
    tktext = nltk.regexp_tokenize(text, pattern)
    voca = sorted(list(set(tktext)))
    len_voca = len(voca)

    print(''.join([str(x) for x in tktext])[:1000])
    #exit()

    input_tokens = []
    output_tokens = []

    for i in range(0, len(tktext) - max_len, step):
        input_tokens.append(tktext[i:i+max_len])
        output_tokens.append(tktext[i+max_len])
    
    # create a 3d table: batch x historic tokens x voca len
    train_data = np.zeros((len(input_tokens), max_len, len_voca))
    target_data = np.zeros((len(input_tokens), len_voca))

    for i , a_slice in enumerate(input_tokens):
        for j, token in enumerate(a_slice):
            train_data[i, j, voca.index(token)] = 1
        target_data[i, voca.index(output_tokens[i])] = 1
    return train_data, target_data, voca, len_voca


def rnn(x, weight, bias, len_voca):
    '''
     define rnn cell and prediction
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, len_voca])
    x = tf.split(x, max_len, 0)

    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.add(tf.matmul(outputs[-1], weight), bias, name='prediction')
    return prediction

def sample(predicted):
    '''
     helper function to sample an index from a probability array
    '''
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

def restore(model):
    
    return

def run(train_data, target_data, voca, len_voca):
    '''
     main run function
    '''

    ####################################################
    parser = OptionParser()
    parser.add_option("-n", "--num-model", dest="MODEL_NUM",
                      help="The Model Number to User")
    parser.add_option("-b", "--num-batch", dest="BATCH_NUM",
                      help="The Batch Number to Use for seed")

    (options, args) = parser.parse_args()
    print(options, args)
    if options.MODEL_NUM == None:
        print("Give -n Model Number (0 to re-initialize training)")
        exit()
    MODEL_NUM = options.MODEL_NUM
    MODEL_PREFIX = "textgen/my_test_model-{}.meta"

    if options.BATCH_NUM == None:
        print("Give -b Batch Number")
        batch_num = 0
    else:
        BATCH_NUM = int(options.BATCH_NUM)

    if int(options.MODEL_NUM) == 0:
        exit()

    else:
        sess = tf.Session()
        graph = tf.get_default_graph()
        # restore variables and tensors
        model = 'textgen/my_test_model-' + options.MODEL_NUM
        saver = tf.train.import_meta_graph(model + '.meta')
        saver.restore(sess, model)
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        bias = graph.get_tensor_by_name("bias:0")
        prediction = tf.get_collection('prediction')[0]
        optimizer = tf.get_collection('optimizer')[0]
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model)
        print("Restored bias:", bias.eval(session=sess))
        
    num_batches = int(len(train_data)/batch_size)
    model_num = int(options.MODEL_NUM)
    #tf.add_to_collection('cost', cost)
    tf.add_to_collection('optimizer', optimizer)
    tf.add_to_collection('prediction', prediction)

    print("----------- Epoch {0} -----------".format(model_num))
    count = BATCH_NUM * batch_size
    train_batch = train_data[count:count+batch_size]

    # Prediction
    #get on of training set as seed
    seed = train_batch[:1:]

    #to print the seed 40 characters
    seed_tokens = ''
    for each in seed[0]:
        seed_tokens += voca[np.where(each == max(each))[0][0]]
    #seed_chars = 'The key objective with the 5G system is '
    print("Seed:", seed_tokens)

    #predict next 1000 tokens
    for i in range(1000):
        if i > 0:
            remove_first_token = seed[:,1:,:]
            seed = np.append(remove_first_token, np.reshape(probabilities, [1, 1, len_voca]), axis=1)
        predicted = sess.run([prediction], feed_dict = {x:seed})
        predicted = np.asarray(predicted[0]).astype('float64')[0]
        probabilities = sample(predicted)
        predicted_token = voca[np.argmax(probabilities)]
        seed_tokens += predicted_token
    print('Result:', seed_tokens)
    sess.close()

if __name__ == "__main__":
    #get data from https://s3.amazonaws.com/text-datasets/nietzsche.txt
    text = read_data('22261-g20.txt')
    train_data, target_data, vocabulary, len_voca = featurize(text)
    print("{0} tokens \nFirst 100 tokens: {1}".format(len_voca, vocabulary[:100]))
    #print("train_data batch",train_data[0])
    #print("target_data",target_data)

    #print("keep some chars",filter(read_data("sample.txt")))
    run(train_data, target_data, vocabulary, len_voca)

