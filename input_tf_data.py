# -*- coding:utf-8 -*-
import  tensorflow as tf
import scipy.io as sio
import  numpy as np
import random
import os
import re
import sys



flags = tf.app.flags
flags.DEFINE_string('output_dir','F:/1硬盘备份/u盘/1论文-风险分析/论文/risk4pycode/cnn_complex_valued/data/tfdata_classify_complex/','tf Dir')
flags.DEFINE_string('data_dir','F:/1硬盘备份/u盘/1论文-风险分析/论文/risk4pycode/cnn_complex_valued/data/data_input/','data dir')
flags.DEFINE_string('label_dir','F:/1硬盘备份/u盘/1论文-风险分析/论文/risk4pycode/cnn_complex_valued/data/data_out_classify/','c dir')
flags.DEFINE_integer('random_seed',0, 'This is the random_seed')
flags.DEFINE_integer('test_num',500,'num of test set')
FLAGS = flags.FLAGS
# data_dir = 'F:/1硬盘备份/u盘/1论文-风险分析/论文/risk4pycode/cnn4case1/data/data_input/'
# label_dir = 'F:/1硬盘备份/u盘/1论文-风险分析/论文/risk4pycode/cnn4case1/data/data_output_classify/'





def read_mat_data4classify(x_dir, y_dir):
    x_data_filename_list = []
    y_data_class_list = []
    y_data_class_dict = sio.loadmat(y_dir+'class_n.mat')
    y_data_class_list_temp = y_data_class_dict['class_n']
    for file in os.listdir(x_dir):
        name = re.split(r'[_.]', file)
        print(name)
        index_seq = int(name[1])
        print(index_seq)
        print(y_data_class_list_temp[index_seq-1][0])
        y_data_class_list.append(y_data_class_list_temp[index_seq-1][0])
        x_data_filename_list.append(x_dir+file)
    print('There is %d samples' % len(x_data_filename_list))
    for i in range(3000):
        print(x_data_filename_list[i])
    print(x_data_filename_list)
    print(y_data_class_list)
    return x_data_filename_list,y_data_class_list

def data_to_tfexample(x_data_single_real,x_data_single_imag,y_data_single,data_shape):
    example =tf.train.Example(features=tf.train.Features(feature={
        'risk_data_real':tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_data_single_real])),
        'risk_data_imag': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_data_single_imag])),
        'data_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=data_shape)),#shape这里别加[]，嗯
        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[y_data_single]))
    }))
    return example

def convert_dataset(split_name, filenames, label_list):
    assert split_name in ['train','test']

    with tf.Session() as sess:
        output_filename = os.path.join(FLAGS.output_dir,split_name+'.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>>Converting risk data %d/%d' %(i+1,len(filenames)))
                    sys.stdout.flush()

                    #loading data
                    var_dict = sio.loadmat(filename)
                    var_name_list = re.split('[/.]',filename)
                    var_name = var_name_list[-2]
                    var_data = var_dict[var_name]
                    var_data = var_data.reshape(33,33,2)
                    var_data_real = var_data.real
                    var_data_imag = var_data.imag
                    # change to bytes
                    print(var_data_real.shape)
                    print(var_data_imag.shape)
                    print(var_data_real.dtype)
                    print(var_data_imag.dtype)##check the dtype  for decode in trainning function
                    var_shape = var_data.shape
                    # var_shape = var_shape.tobytes()
                    var_data_real = var_data_real.tobytes()
                    var_data_imag = var_data_imag.tobytes()
                    var_label = label_list[i]
                    #生成protocol数据类型
                    example = data_to_tfexample(var_data_real,var_data_imag, var_label, var_shape)
                    tfrecord_writer.write(example.SerializeToString())

                except IOError as e:
                    print('Could not read risk data', filename)
                    print('Error', e)
    sys.stdout.write('\n')
    sys.stdout.flush()

def main(unuse_args):
    filenames_list,labels_list = read_mat_data4classify(FLAGS.data_dir, FLAGS.label_dir)

    #把数据切分为训练集和测试集, 并打乱
    random.seed(FLAGS.random_seed)
    random.shuffle(filenames_list)
    random.seed(FLAGS.random_seed)
    random.shuffle(labels_list)
    training_filenames = filenames_list[FLAGS.test_num:]
    training_labelslist = labels_list[FLAGS.test_num:]
    testing_filenames = filenames_list[:FLAGS.test_num]
    testing_labelslist = labels_list[:FLAGS.test_num]

    # 数据转换
    convert_dataset('train', training_filenames, training_labelslist)
    convert_dataset('test', testing_filenames, testing_labelslist)
    print('Finish!!!!!!!!!!!!!!!!!')

if __name__ == '__main__':
    tf.app.run()





