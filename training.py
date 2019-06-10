# %%

import os
import numpy as np
import tensorflow as tf
import input_tf_data
import model

# %%



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_tfrecord_dir','F:/1硬盘备份/u盘/1论文-风险分析/论文/risk4pycode/cnn_complex_valued/data/tfdata_classify_complex/train.tfrecords','dir')
tf.app.flags.DEFINE_string('validate_tfrecord_dir','F:/1硬盘备份/u盘/1论文-风险分析/论文/risk4pycode/cnn_complex_valued/data/tfdata_classify_complex/test.tfrecords','dir')
tf.app.flags.DEFINE_string('logs_train_dir','F:/1硬盘备份/u盘/1论文-风险分析/论文/risk4pycode/cnn_complex_valued/result_log4classify/train','LOGS')
tf.app.flags.DEFINE_string('logs_val_dir','F:/1硬盘备份/u盘/1论文-风险分析/论文/risk4pycode/cnn_complex_valued/result_log4classify/val','LOGS')
tf.app.flags.DEFINE_integer('batch_size',20,'size')
tf.app.flags.DEFINE_integer('n_classes',4,'num of class')
tf.app.flags.DEFINE_integer('print_interval',10,'interval')
tf.app.flags.DEFINE_float('learning_rate',0.01,'rate')
tf.app.flags.DEFINE_integer('max_step',3000,'num of steps')


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serializerd_example = reader.read(filename_queue)
    features = tf.parse_single_example(serializerd_example,
                                       features={
                                           'risk_data_real':tf.FixedLenFeature([],tf.string),
                                           'risk_data_imag': tf.FixedLenFeature([], tf.string),
                                           'data_shape':tf.FixedLenFeature([3],tf.int64),
                                           'label':tf.FixedLenFeature([1],tf.int64)
                                       })
    #获取数据
    risk_data_real = tf.decode_raw(features['risk_data_real'],tf.float64)
    risk_data_imag = tf.decode_raw(features['risk_data_imag'], tf.float64)
    risk_data = tf.complex(risk_data_real,risk_data_imag)
    data_shape = tf.cast(features['data_shape'],tf.int32)
    risk_data = tf.reshape(risk_data,shape=[33,33,2])
    #获取label
    label = tf.cast(features['label'],tf.int32)
    label = tf.reshape(label,shape=[1])

    return risk_data, label


x = tf.placeholder(tf.complex64,shape=[FLAGS.batch_size,33,33,2])
y_ = tf.placeholder(tf.int32,shape=[FLAGS.batch_size])
is_training = tf.placeholder(dtype=tf.bool)

#loading train data
tra_risk_data, tra_label = read_and_decode(FLAGS.train_tfrecord_dir)
tra_risk_data_batch, tra_label_batch = tf.train.shuffle_batch([tra_risk_data, tra_label],
                                                      batch_size=FLAGS.batch_size,
                                                      capacity=50000,
                                                      min_after_dequeue=10000,
                                                      num_threads=1)
tra_label_batch = tf.reshape(tra_label_batch,[FLAGS.batch_size])

#loading test data
val_risk_data, val_label = read_and_decode(FLAGS.validate_tfrecord_dir)
val_risk_data_batch, val_label_batch = tf.train.shuffle_batch([val_risk_data, val_label],
                                                      batch_size=FLAGS.batch_size,
                                                      capacity=50000,
                                                      min_after_dequeue=10000,
                                                      num_threads=1)
val_label_batch = tf.reshape(val_label_batch,[FLAGS.batch_size])
#set training op
train_logits = model.inference(x,batch_size=FLAGS.batch_size,n_classes=FLAGS.n_classes,is_training=is_training)
train_loss = model.losses(train_logits,y_)
train_op = model.trainning(train_loss, learning_rate=FLAGS.learning_rate)
train_acc = model.evaluation(train_logits,y_)


#保存模型op
saver = tf.train.Saver()

def main(unuse_args):
    with tf.Session() as sess:
        #init
        sess.run(tf.global_variables_initializer())
        #创建协调器 管理线程,启动读取队列,不启动则为空白队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.logs_val_dir, sess.graph)
        #BEGIIN!!!!!!!!!!!!!!!!!!!!!!!
        try:
            for step in np.arange(FLAGS.max_step):
                if coord.should_stop():
                    break

                tra_datas, tra_labels = sess.run([tra_risk_data_batch, tra_label_batch])
                # for i in range(FLAGS.batch_size):
                #     print(tra_datas[i],tra_labels[i])
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
                                                feed_dict={x: tra_datas, y_: tra_labels, is_training:True})
                if step % 20 == 0 or (step + 1) == FLAGS.max_step:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0),end='')
                    summary_str = sess.run(summary_op,
                                           feed_dict={x: tra_datas, y_: tra_labels,is_training:True})
                    train_writer.add_summary(summary_str, step)

                    # if step % 100 == 0 or (step + 1) == MAX_STEP:
                    val_datas, val_labels = sess.run([val_risk_data_batch, val_label_batch])
                    val_loss, val_acc = sess.run([train_loss, train_acc],
                                                 feed_dict={x: val_datas, y_: val_labels,is_training:False})
                    print('**Step %d, val loss = %.2f, val accuracy = %.2f%% **' % (step, val_loss, val_acc * 100.0))
                    summary_str = sess.run(summary_op,
                                           feed_dict={x: val_datas, y_: val_labels,is_training:False})
                    val_writer.add_summary(summary_str, step)
                    checkpoint_path = os.path.join(FLAGS.logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                    # if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    #     checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    #     saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
