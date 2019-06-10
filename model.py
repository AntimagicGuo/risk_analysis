# %%
import tensorflow as tf
#def c_con2d


def complex_conv2d(input,fliter,strides,padding):
    '''complex valued conv2d
    Args:
         input: complex tensor,[batch,height,width,channnel]
         fliter: complex tensor,[height,width,in_channel,out_channel]
        strides:[1,height_slide,width_slide,1]
        padding:'SAME'or'VALID'
    return:
        output:complex tensor,shape=[]
    '''
    # input=A+Bi
    # f=x+yi
    input_real = tf.real(input)
    input_imag = tf.imag(input)
    fliter_real = tf.real(fliter)
    fliter_imag = tf.imag(fliter)
    output_real = tf.nn.conv2d(input_real,fliter_real,strides,padding) - tf.nn.conv2d(input_imag,fliter_imag,strides,padding)
    output_imag = tf.nn.conv2d(input_real,fliter_imag,strides,padding) + tf.nn.conv2d(input_imag,fliter_real,strides,padding)
    output = tf.complex(output_real,output_imag)

    return output

def c_batch_norma_layer(input,is_training=False,name='batch_norm_c'):
    '''
    Args:
        input:complex value
    Return:
        batch norm
    '''
    input_real = tf.real(input)
    input_imag = tf.imag(input)
    if is_training is True:# 训练模式 使用指数加权函数不断更新均值和方差
        output_real = tf.contrib.layers.batch_norm(inputs=input_real,decay=0.9,updates_collections=None,is_training=True)
        output_imag = tf.contrib.layers.batch_norm(inputs=input_real,decay=0.9,updates_collections=None,is_training=True)
    else: # ;测试模式  不更新均值和方差，直接使用
        output_real = tf.contrib.layers.batch_norm(inputs=input_real,decay=0.9,updates_collections=None,is_training=False)
        output_imag = tf.contrib.layers.batch_norm(inputs=input_real,decay=0.9,updates_collections=None,is_training=False)
    output=tf.complex(output_real,output_imag)

    return  output

def batch_norma_layer(input,is_training=False,name='batch_norm_c'):
    '''
    Args:
        input:real value
    Return:
        batch norm
    '''
    if is_training is True:# 训练模式 使用指数加权函数不断更新均值和方差
        output = tf.contrib.layers.batch_norm(inputs=input,decay=0.9,updates_collections=None,is_training=True)
    else: # ;测试模式  不更新均值和方差，直接使用
        output = tf.contrib.layers.batch_norm(inputs=input, decay=0.9, updates_collections=None, is_training=True)

    return  output

def zRelu(input):

    '''
    Args:
        input:complex tensor
    Return;
         output
    '''
    input_real = tf.real(input)
    input_imag = tf.imag(input)
    flag_real = tf.cast(tf.cast(input_real,tf.bool),tf.float32)
    flag_imag = tf.cast(tf.cast(input_imag,tf.bool),tf.float32)
    flag = flag_real*flag_imag
    output_real = input_real*flag
    output_imag = input_imag*flag
    output = tf.complex(output_real,output_imag)

    return output


def cRelu(input):
    '''
    Args:
        input: complex tensor
    Return:
        output
    '''
    input_real = tf.real(input)
    input_imag = tf.imag(input)
    output_real = tf.nn.relu(input_real)
    output_imag = tf.nn.relu(input_imag)
    output = tf.complex(output_real,output_imag)

    return output

def c_avg_pool(input,ksize,strides,padding):
    '''
    Args;
         input: complex tensor
         ksize: [1,k,k,1]
        strides: [1,s,s,1]
        padding:same or valid
    return:
        output
    '''
    input_real = tf.real(input)
    input_imag = tf.imag(input)
    output_real = tf.nn.avg_pool(input_real,ksize,strides,padding)
    output_imag = tf.nn.avg_pool(input_imag,ksize,strides,padding)
    output = tf.complex(output_real,output_imag)

    return output

def c_max_pool(input, ksize, strides, padding):
    '''
    Args;
         input: complex tensor
         ksize: [1,k,k,1]
        strides: [1,s,s,1]
        padding:same or valid
    return:
        output
    '''
    input_real = tf.real(input)
    input_imag = tf.imag(input)
    output_real = tf.nn.max_pool(input_real, ksize, strides, padding)
    output_imag = tf.nn.max_pool(input_imag, ksize, strides, padding)
    output = tf.complex(output_real, output_imag)

    return output


def complex_to_real(input):
    input_real = tf.real(input)
    input_imag = tf.imag(input)
    output = tf.sqrt(tf.add(tf.square(input_real), tf.square(input_imag)))

    return output


# %%
def inference(images, batch_size, n_classes,is_training):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    with tf.variable_scope('conv1') as scope:
        weights_real = tf.get_variable('weights_real',
                                  shape=[3, 3, 2, 16],#16个卷积核
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.33, dtype=tf.float32))
        # weights_real = tf.abs(weights_real)
        weights_imag = tf.get_variable('weights_imag',
                                  shape=[3, 3, 2, 16],#16个卷积核
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.33, dtype=tf.float32))
        # weights_imag = tf.abs(weights_imag)
        weights = tf.complex(weights_real,weights_imag,name='weights')
        biases_real = tf.get_variable('biases_real',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # biases_real = tf.abs(biases_real)
        biases_imag = tf.get_variable('biases_imag',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # biases_imag = tf.abs(biases_imag)
        biases = tf.complex(biases_real,biases_imag,name='biases')
        conv = complex_conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        batch_normal = c_batch_norma_layer(pre_activation,is_training=is_training)
        conv1 = zRelu(batch_normal)

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = c_max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
        norm1 = pool1

    # conv2
    # with tf.variable_scope('conv2') as scope:
    #     weights_real = tf.get_variable('weights_real',
    #                               shape=[3, 3, 16, 16],
    #                               dtype=tf.float32,
    #                               initializer=tf.truncated_normal_initializer(stddev=0.118, dtype=tf.float32))
    #     # weights_real = tf.abs(weights_real)
    #     weights_imag = tf.get_variable('weights_imag',
    #                               shape=[3, 3, 16, 16],
    #                               dtype=tf.float32,
    #                               initializer=tf.truncated_normal_initializer(stddev=0.118, dtype=tf.float32))
    #     # weights_imag = tf.abs(weights_imag)
    #     weights = tf.complex(weights_real,weights_imag,name='weights')
    #     biases_real = tf.get_variable('biases_real',
    #                              shape=[16],
    #                              dtype=tf.float32,
    #                              initializer=tf.constant_initializer(0.1))
    #     # biases_real = tf.abs(biases_real)
    #     biases_imag = tf.get_variable('biases_imag',
    #                              shape=[16],
    #                              dtype=tf.float32,
    #                              initializer=tf.constant_initializer(0.1))
    #     # biases_imag = tf.abs(biases_imag)
    #     biases = tf.complex(biases_real,biases_imag,name='biases')
    #     conv = complex_conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     batch_normal = c_batch_norma_layer(pre_activation)
    #     conv2 = zRelu(batch_normal)
    #
    # # pool2 and norm2
    # with tf.variable_scope('pooling2_lrn') as scope:
    #     norm2 = conv2
    #     pool2 = c_max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
    #                            padding='SAME')

    # local3
    with tf.variable_scope('local3') as scope:
        pool2 = complex_to_real(pool1)
        reshape_op = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape_op.get_shape()[1].value#
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        batch_norm = batch_norma_layer(tf.matmul(reshape_op, weights) + biases,is_training=is_training)
        local3 = tf.nn.relu(batch_norm, name=scope.name)

        # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.125, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        batch_norm = batch_norma_layer(tf.matmul(local3, weights) + biases,is_training=is_training)
        local4 = tf.nn.relu(batch_norm, name='local4')

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.125, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights),biases, name='softmax_linear')

    return softmax_linear


# %%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits+1e-5, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# %%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # threshold = 1.0
        # grads_and_vars = optimizer.compute_gradients(loss)
        # capped_gvs = [(tf.clip_by_value(grad,-threshold,threshold),var) for grad,var in grads_and_vars]
        train_op = optimizer.minimize(loss, global_step=global_step)
        # train_op =optimizer.apply_gradients(capped_gvs)
    return train_op


# %%
def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy





# test
# real =  tf.get_variable('weights_real',
#                         shape=[3, 3, 3, 16],#16个卷积核
#                         dtype=tf.float32,
#                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
# imag =  tf.get_variable('weights_imag',
#                         shape=[3, 3, 3, 16],#16个卷积核
#                         dtype=tf.float32,
#                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
# weight = tf.complex(real,imag,'weight_complex')
# sess = tf.InteractiveSession()
# init = sess.run(tf.global_variables_initializer())
# # print(real.eval())
# # print(imag.eval())
# # print(weight.eval())
# x = tf.constant([1,2,3],dtype=tf.float32)
# y = tf.constant([4,5,6],dtype=tf.float32)
# c = tf.complex(x,y)
# print(c.eval())
