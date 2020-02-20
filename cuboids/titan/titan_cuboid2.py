


# In[1]:

from __future__ import division, print_function
import imageio
imageio.plugins.ffmpeg.download()
from random import shuffle as sf
from itertools import izip as zip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pims
import numpy as np
import os
import glob
import scipy.stats as st
#import conv_cosnorm as cos
#59% in test

class paramethers:
    def __init__(self):
        self.is_training = True
        
        #Input parameters
        self.batch_size = 96
        self.rows, self.cols, self.channels, self.depth= 100, 100, 1, 40
        self.n_classes = 20
        
        #Trainning parameters
        self.initial_learning_rate =3e-3
        self.total_train =   6822 #9575
        self.total_validation = 2753
        self.total_test = 3574
        self.decay_steps =1e3
        self.decay_rate = 0.99
        self.epochs = 1000 if self.is_training  else 1
        self.steps_val = self.total_validation // self.batch_size
        self.max_steps = (self.total_train//self.batch_size)* self.epochs if self.is_training  else (self.total_test//self.batch_size + 1)
        self.summary_step = 50
        self.keep_dropout = .91
        self.inception_layers = 4

        #Model parameters
        self.kernels_size = [5, 9, 11]
        self.kernel_size = 6
        self.layer_depth = 1024
        self.spacial_filters_size = 64 
        self.first_temp_filters_size = 8
        self.temporal_kernel_size = 5
        self.filters_size = 2
        self.temp_kernel = 1
        self.reg_scale = 0.01
        self.vggpool=3
        self.tfrecord_base_name="last_{}_sampled_sub".format(self.depth)
        self.tfrecord_base_name_test="last_{}_sampled_sub".format(self.depth)
        self.vgg_layer_name = "import/pool{}:0".format(self.vggpool)
        self.save_checkpoint_secs = 600
        self.regularize_type = 2
       
        
        
class inception_cuboid:
    def __init__(self, param = None):
        
        self.param = paramethers() if param is None else param
        self.sess = None
        
       
    def __create_cuboid_vgg_model(self):
       
        print("Batch size: ",self.param.batch_size)
        self.prediction, self.optimizer, self.loss, self.accuracy = self. cuboid_vgg_model()

        print("tensorboard --logdir={}\n\n".format(self.checkpoint_dir))

        iter_per_epoch = self.param.max_steps//self.param.epochs
        print("iter per epoch:",iter_per_epoch)
        def formatter(d): 
            return ' epoch {:d}, loss: {:.2f},  train_acc: {:.2f}%'.format(d.values()[2]//iter_per_epoch, d.values()[0], d.values()[1] * 100)

        # with tf.device('/gpu:0'):
        hooks = [
            tf.train.StopAtStepHook(num_steps=self.param.max_steps),
            tf.train.LoggingTensorHook([self.loss, self.accuracy, self.global_step], every_n_iter=iter_per_epoch, formatter=formatter),
        #     tf.train.NanTensorHook(loss),
        ]

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 8
        config.inter_op_parallelism_threads = 8

        self.sess = tf.train.MonitoredTrainingSession(checkpoint_dir=self.checkpoint_dir,
                                            hooks=hooks,
                                            save_checkpoint_secs=self.param.save_checkpoint_secs,
                                            save_summaries_steps=self.param.summary_step,
                                            config=config)



    def __create_inseption_model(self):
        print("Batch size: ",self.param.batch_size)
        self.prediction, self.optimizer, self.loss, self.accuracy = self.model()

        print("tensorboard --logdir={}\n\n".format(self.checkpoint_dir))

        iter_per_epoch = self.param.max_steps//self.param.epochs
        print("iter per epoch:",iter_per_epoch)
        def formatter(d): 
            return ' epoch {:d}, loss: {:.2f},  train_acc: {:.2f}%'.format(d.values()[2]//iter_per_epoch, d.values()[0], d.values()[1] * 100)

        # with tf.device('/gpu:0'):
        hooks = [
            tf.train.StopAtStepHook(num_steps=self.param.max_steps),
            tf.train.LoggingTensorHook([self.loss, self.accuracy, self.global_step], every_n_iter=iter_per_epoch, formatter=formatter),
        #     tf.train.NanTensorHook(loss),
        ]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44

        self.sess = tf.train.MonitoredTrainingSession(checkpoint_dir=self.checkpoint_dir,
                                            hooks=hooks,
                                            save_checkpoint_secs=self.param.save_checkpoint_secs,
                                            save_summaries_steps=self.param.summary_step,
                                            config=config)



        
    def configure(self):
        tf.set_random_seed(2018)
        self.keep = tf.placeholder(tf.float32)
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(self.param.initial_learning_rate, self.global_step, 
                                                self.param.decay_steps, self.param.decay_rate, staircase=True)

        tf.summary.scalar("learning_rate", self.learning_rate)
        self.tfrecord_input_file = '/notebooks/video_tfrecord/video_{}_{}.tfrecords'.format( "train" if self.param.is_training  else "test", self.param.tfrecord_base_name if self.param.is_training  else self.param.tfrecord_base_name_test)
        self.checkpoint_dir="/notebooks/cuboids/checkpoints/cub-vgg_{}_D{}_FTFS-{}_FS-{}_SFS-{}_TKS-{}_KS-{}_pool{}_LR-{:.0E}_drop-{}_B{}".format(self.param.tfrecord_base_name,
                                                                                                                             self.param.layer_depth,
                                                                                                                             self.param.first_temp_filters_size,
                                                                                                                             self.param.filters_size, 
                                                                                                                             self.param.spacial_filters_size,
                                                                                                                             self.param.temporal_kernel_size, 
                                                                                                                             self.param.kernel_size,
                                                                                                                             self.param.vggpool, 
                                                                                                                             self.param.initial_learning_rate, 
                                                                                                                             self.param.keep_dropout, 
                                                                                                                             self.param.batch_size )

        with tf.device('/cpu:0'):
            with tf.name_scope('video_input'):
                self.input_tensor, self.labels = self.__load_inputs(self.tfrecord_input_file, self.param.batch_size)

        print("input file:", self.tfrecord_input_file)
        print("Check point dir: ", self.checkpoint_dir)
        
        #self.__create_inseption_model()
        self.__create_cuboid_vgg_model()


        
    


    def temporal_conv(self, tensor, kernel_size, channels, num_filters, layer_id = 0):   
        if len(tensor.shape) == 4:
            input = tf.expand_dims(tensor, 1)
        else: input = tensor

        filters, bias = self.get_temp_filters(kernel_size, channels, num_filters, 1, id)
        
        temp_layer = tf.nn.conv3d(input, filters, [1,1,1,1,1], data_format= "NDHWC",padding='VALID', name="temp_conv")
    #     temp_layer = cos.conv3d_cosnorm(input, filters, strides=[1,1,1,1,1], padding='VALID')
        temp_layer = tf.nn.bias_add(temp_layer, bias)
    #     temp_layer =  batch_normalize(temp_layer)
        
        temp_layer = tf.nn.relu(temp_layer)
        return temp_layer

    def get_temp_filters(self, temporal_kernel_size, channels, temp_filters_size, spacial_kernel_size=1, id = 0):
            filter = tf.Variable(tf.random_normal([ temporal_kernel_size, spacial_kernel_size, spacial_kernel_size,   channels, temp_filters_size ]), 
                            dtype=tf.float32, name="temp_filters_")
            bias = tf.Variable(tf.random_normal([temp_filters_size]), name="B_temp")
            return filter, bias
    
    def get_fc_weights(self, w_inputs, w_output, id=0):
        
        weight= tf.Variable(tf.truncated_normal([w_inputs, w_output]), name="{}/weight".format(id))
        bias =  tf.Variable(tf.truncated_normal([w_output]), name="{}/bias".format(id))
        return weight, bias


    def get_spacial_filter(self, kernel_size, name = "gauss"):
        filter1= tf.Variable(tf.random_normal([1, kernel_size, 1, 1]), 
                                    dtype=tf.float32, name="W1_{}".format(name))
        filter2= tf.Variable(tf.random_normal([ kernel_size, 1, 1, 1]), 
                                    dtype=tf.float32, name="W2_{}".format(name))
                    
        bias = tf.Variable(tf.random_normal([1]), name="B_{}".format(name))
        return filter1, filter2, bias

    def get_conv2d_filters(self, kernel_size, depth=1, name = "cuboid"):
        filter= tf.Variable(tf.random_normal([kernel_size, kernel_size, depth, 3]), 
                                    dtype=tf.float32, name="W_{}".format(name))
        bias = tf.Variable(tf.random_normal([3]), name="B_{}".format(name)) 
        return filter, bias


    def fully_conn(self, previous_layer, in_size, out_size, keep, layer_id):
        print("drop out:", keep)
        w, b = self.get_fc_weights(in_size, out_size, layer_id)
        fc_layer = tf.nn.bias_add(tf.matmul(previous_layer, w), b, name="{}/matmul-add".format(layer_id))
    #     fc_layer = tf.nn.bias_add(cos.fc_cosnorm(previous_layer, weight), bias, name="{}/matmul-add".format(layer_id))
        fc_layer = tf.nn.relu(fc_layer, name="{}/relu".format(layer_id))
        fc_layer = tf.nn.dropout(fc_layer, keep, name="{}/dropout".format(layer_id))
        fc_layer = self.batch_norm(fc_layer, scope = "BN_fc/{}".format(layer_id))
        return fc_layer

    def conv_spac_tensor(self, input, kernel_size=3):
        input=tf.expand_dims(input,-1)
        input = tf.transpose(input, perm=[0, 3, 1,2,4])
        
        filters, bias = self.get_temp_filters( temporal_kernel_size=1, channels=1, temp_filters_size=1, spacial_kernel_size=kernel_size, id = 0)
        print("Filter:",filters)
        print("input:",input)
        layer = tf.nn.conv3d(input, filters, [1,1,1,1,1], data_format= "NDHWC",padding='VALID', name="smooth_conv")
        layer = tf.nn.bias_add(layer, bias)
        
        layer = tf.nn.relu(layer)
        layer = tf.squeeze(tf.transpose(layer, perm=[0, 4, 2,3,1]))
        return layer


    def conv_tensor(self, tensor, kernel_size=3, depth=1, strides=1, id=0):
            filter1, filter2, bias = self.get_spacial_filter(kernel_size)
            
            conv_tensor = []
            
            images = tf.unstack(tf.squeeze(tensor), axis=3)
            for id, image in zip(range(len(images)), [tf.expand_dims(img,-1) for img in images]) :
                conv_image = tf.nn.conv2d(image, filter1, [1,1,1,1],"SAME",  name = "conv2d_temp")
                conv_image = tf.nn.conv2d(conv_image, filter2, [1,1,1,1],"SAME",  name = "conv2d_temp")
                conv_image = tf.nn.bias_add(conv_image, bias)
    #             conv_image =tf.transpose(conv_image,[0,3,1,2])
                images[id]=conv_image
            tensor = tf.squeeze(tf.stack( images, axis=3))
            
            
           
    #         x =  tf.contrib.layers.batch_norm(x,center=True, scale=True, is_training=not is_test)
    #         x =  tf.contrib.layers.batch_norm(x)
            return tensor


    def maxpool(self, tensor, k=2, d=1):
        if len(tensor.shape) == 5:
            tensor = tf.transpose(tensor, perm=[0, 3, 1,2,4])
            return tf.nn.max_pool3d(tensor, ksize=[1,  k,k, k, d], strides=[1, k, k, k, d], data_format= "NDHWC" , padding='VALID')
        
        return tf.nn.max_pool(tensor, ksize=[1, k, k, d], strides=[1, k, k, d], padding='VALID' )
                        
                            
    def tensor_maxpool(self, tensor, axis = 4, layer_id = 0):
        if len(tensor.shape) == 4:
            tensor=tf.expand_dims(tensor,1)
        tensor_mp = tf.reduce_max(tensor, axis=axis, keep_dims = True)
        return tensor_mp

    def tensor_sum(self, tensor, axis = 4, layer_id = 0):
        tensor_sum = tf.reduce_sum(tensor, axis=axis, keep_dims = True)
    #     tensor_mp =  batch_norm(tensor_mp, scope = "BN_temp/{}".format(layer_id))
        return tensor_sum

    def tensor_avgpool(self, tensor, axis = 4, layer_id = 0):
        tensor_ap = tf.reduce_mean(tensor, axis=axis)
        
    #     tensor_ap =  batch_norm(tensor_ap, scope = "BN_space/{}".format(layer_id))
        return tensor_ap


    def cuboid_model(self, tensor, kernel_size,  num_filters, id, spac_kernel=3):
    #     with tf.device('/cpu:0'):
        # if len(tensor.shape) == 5:
        #     input = tf.transpose(tensor, perm=[0, 4, 2,3,1])
        # else:
        #     input=tf.expand_dims(tensor,-1)
        #     input = tf.transpose(input, perm=[0, 4, 2,3,1])
        input=tensor
        print(input.shape)
        channels = 1
        #Spacial convolution. (Behave as a gaussian. it is aplied to each image in de video.)
       # smoothed = self.conv_tensor(input,kernel_size)
        smoothed = self.conv_spac_tensor(input, spac_kernel)
        print("smoothed:", smoothed)

        smoothed=tf.expand_dims(smoothed,-1)
        smoothed = tf.transpose(smoothed, perm=[0, 3, 1,2,4])
        
        #Temporal convolution (such as quadrature pair filters)
        quadracture_pair  = self.temporal_conv(smoothed, kernel_size, channels, num_filters, layer_id = id)
        
        
        #Sum of the quadrature pair
        quadracture_pair_sum = self.tensor_sum(quadracture_pair, layer_id=id)
        cuboid_layer = tf.squeeze(tf.transpose(quadracture_pair_sum, perm=[0, 4, 2,3,1]))
        print("cuboid", cuboid_layer )
        # cuboid_layer =  rgb_representation(cuboid_layer);
        return cuboid_layer

    def cuboid_vgg_model(self):
        input = tf.squeeze(self.input_tensor)
        input = self.batch_norm(input, "BN_input")
        print(input)

        with tf.name_scope('cuboid1'):
            #cuboid= self.inseptinon_layer(input, self.param.channels, self.param.kernels_size, "1")
            cuboid= self.cuboid_model(input, 9, 2, 1, 5)
            cuboid = self.maxpool(tf.expand_dims(cuboid,-1))
            cuboid = tf.transpose(tf.squeeze(cuboid), perm=[0,2,3,1])
            cuboid = self.batch_norm(cuboid, "BN_cuboid1")


        with tf.name_scope('cuboid2'):
            cuboid= self.cuboid_model(cuboid, 5, 2, 2,3)
            cuboid = self.maxpool(tf.expand_dims(cuboid,-1))
            cuboid = tf.transpose(tf.squeeze(cuboid), perm=[0,2,3,1])
            cuboid = self.batch_norm(cuboid, "BN_cuboid2")

            

        with tf.name_scope('cuboid3'):
            cuboid= self.cuboid_model(cuboid, 2, 2, 1, 1)
            cuboid = self.maxpool(tf.expand_dims(cuboid,-1))
            cuboid = tf.transpose(tf.squeeze(cuboid), perm=[0,2,3,1])
            cuboid = self.batch_norm(cuboid, "BN_cuboid3")


        #rgb = cuboid
        print("cuboid out:",cuboid)
        # with tf.name_scope('Batch_Norm_Cub'):
        #     cuboid = self.batch_norm(cuboid, "BN_cuboid_out")
        # with tf.name_scope('spatial_conv'):
        #     rgb = self.conv(cuboid, 5)
        
        # with tf.name_scope('Batch_Norm_rgb'):
        #     rgb = self.batch_norm(rgb, "BN_rgb")
        #     tf.summary.image("images_cuboids", rgb)
        # with tf.name_scope('vgg16'):
        #      vgg,_ = self.__load_trained_model(rgb, 'vgg16')
        #      print("VGG output :", vgg)
        with tf.name_scope('flatten'):
            flatten = tf.layers.flatten(cuboid)
            print("flatten:", flatten)


        
        with tf.name_scope('fully_conn'):
            keep = 0.5 if self.param.is_training else 1.0
            fc = self.fc_model(flatten, keep = keep)
            
        with tf.name_scope('softmax'):
            pred = self.softmax_layer(fc)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels))
            tf.summary.scalar("loss", loss)

        with tf.name_scope('regularized_loss'):
            reg_loss = self.regularize(loss, type = self.param.regularize_type, scale=self.param.reg_scale)
            tf.summary.scalar("regularized", reg_loss)
            

        with tf.name_scope('sgd'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(reg_loss, global_step=self.global_step)

        with tf.name_scope('Train_accuracy'):
            acc = tf.equal(tf.argmax(pred, 1), tf.argmax(self.labels, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            tf.summary.scalar("accuracy", acc)

        return pred, optimizer, reg_loss, acc
        

    #     output = rgb_representation(tf.squeeze(output));
        return output

    def inseptinon_layer(self, tensor, channels, kernels_size, id):
        
        c1= tf.expand_dims(self.cuboid_model(tensor, kernels_size[0], channels, 2, "{}_{}".format(id,1)), -1)
        c2= tf.expand_dims(self.cuboid_model(tensor, kernels_size[1], channels, 2, "{}_{}".format(id,2)), -1)
        c3= tf.expand_dims(self.cuboid_model(tensor, kernels_size[2], channels, 2, "{}_{}".format(id,3)), -1)
        stack=tf.squeeze(tf.stack([c1,c2,c3],4))
        output = self.tensor_avgpool(stack, layer_id=id)
        print("inseption out:", output)

    #     output = rgb_representation(tf.squeeze(output));
        return output

    def conv(self,x, kernel_size):
        depth = x.shape.as_list()[-1]
        W, b = self.get_conv2d_filters(kernel_size, depth)
        x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],data_format= "NHWC", padding='VALID' )
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
        return x 
        
        

    def batch_norm(self, tensor, scope = None):
        return tf.layers.batch_normalization(tensor, name = scope, training = self.param.is_training )

    def rgb_representation(self, tensor):
        max = tf.reduce_max(tensor, axis=3, keep_dims = True)
    #     m, v = tf.nn.moments(tensor, [3], keep_dims = True)
    #     rgb = tf.stack([max, m, v], axis = 3)
        rgb = tf.image.grayscale_to_rgb(max)
        return rgb

    def regularize(self, loss, type = 1,scale = 0.005, scope = None):
        if type == 1:
            regularizer = tf.contrib.layers.l1_regularizer( scale=scale, scope=scope)
        else:
            regularizer = tf.contrib.layers.l2_regularizer( scale=scale, scope=scope)
                
        weights = tf.trainable_variables() # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, weights)
        regularized_loss = loss + regularization_penalty
        return regularized_loss


    def __read_decode_distort(self, queue):
        reader = tf.TFRecordReader()
        _, serialized = reader.read(queue)
        features = tf.parse_single_example(serialized, features={
        'video_raw': tf.FixedLenFeature([self.param.channels * 
                                         self.param.rows *
                                         self.param.cols * 
                                         self.param.depth], 
                                        tf.string),
            
        'label': tf.FixedLenFeature([], tf.int64),
        })
        
        video = tf.decode_raw(features['video_raw'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)
        
        video = tf.cast(video, tf.float32)
        video = tf.reshape(video, [self.param.channels, self.param.rows, self.param.cols, self.param.depth])
        if self.param.is_training:
            video = tf.image.resize_bicubic(video, [90, 90])
            video = tf.random_crop(video, [self.param.channels, 90, 80, self.param.depth])
            video = tf.expand_dims(tf.image.random_flip_left_right(tf.squeeze(video)), 0)

        else:
            video = tf.image.resize_bicubic(video, [90, 80])
        return video, label


    def __load_inputs(self, filename, batch_size):
        queue = tf.train.string_input_producer([filename])

        image, label = self.__read_decode_distort(queue)
        if self.param.is_training:
            image_batch, label_batch = tf.train.shuffle_batch(
                [image, label], 
                batch_size = self.param.batch_size, 
                num_threads = 11, 
                capacity = 70 * self.param.batch_size, 
                min_after_dequeue = 20 * self.param.batch_size)
        else:
            image_batch, label_batch = tf.train.batch(
                [image, label], 
                batch_size = self.param.batch_size, 
                num_threads = 7, 
                capacity = 20 * self.param.batch_size)

        label_batch = tf.one_hot(label_batch, self.param.n_classes)
        return image_batch, label_batch




    def fc_model(self, input, keep = 0.5):    
        out_shape = input.shape.as_list()
        print("Flatten :",out_shape)
        net = self.fully_conn(input, np.prod(out_shape[1:]), self.param.layer_depth, keep, "fc1" )
        # net = self.fully_conn(net, self.param.layer_depth, self.param.layer_depth, keep, "fc2" )
        # net = self.fully_conn(net, self.param.layer_depth, self.param.layer_depth, keep, "fc3" )
        return net

    def softmax_layer(self, fc_layer):
        out_shape = fc_layer.shape.as_list()
        w, b = self.get_fc_weights(np.prod(out_shape[1:]), self.param.n_classes, "softmax")
        net = tf.add(tf.matmul(fc_layer, w), b, name="fc4/matmul-add")
        return net

    def __load_trained_model(self, tensor, name):
        with open("/notebooks/trained_models/vgg16-20160129.tfmodel", mode='rb') as f:
            content = f.read()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(content)

            images = tensor
            
            graph_def = tf.graph_util.extract_sub_graph(graph_def, ["images", "pool5"])
            tf.import_graph_def(graph_def, input_map={"images": images})

        del content
        graph = tf.get_default_graph()
        
        net = graph.get_tensor_by_name(name +"/"+ self.param.vgg_layer_name)
        return net, graph
        



    def model(self):
        input = tf.squeeze(self.input_tensor)
        input = self.batch_norm(input, "BN_input")
        print(input)
            
        # with tf.name_scope('spacial'):
        #     spacial = tf.squeeze(self.conv_tensor(input, 5))
        #     spacial = tf.expand_dims(spacial, -1)
        #     spacial = tf.transpose(spacial, perm=[0, 4, 1, 2,3])
            

        # layer = spacial
        # layer = self.batch_norm(layer, scope = "BN_spacial")
        layer = input
        for i in range(self.param.inception_layers):
            with tf.name_scope('inception_{}_1'.format(i)):
                layer = self.inseptinon_layer(layer, self.param.channels, self.param.kernels_size, i)
            with tf.name_scope('inception_{}_2'.format(i)):
                layer = self.inseptinon_layer(layer, self.param.channels, self.param.kernels_size, i)
            
            with tf.name_scope('maxpool_{}'.format(i)):
                layer = self.maxpool(layer)
                layer = self.batch_norm(layer, scope = "BN_inseptcon_{}".format(i))
                layer = tf.expand_dims(layer,1)

        print(layer.shape.as_list())
        with tf.name_scope('flatten'):
            flatten = tf.layers.flatten(layer, name="flatten")

        
        with tf.name_scope('fully_conn'):
            keep = 0.5 if self.param.is_training else 1.0
            print("keep:", keep)
            fc = self.fc_model(flatten, keep=keep)
            
        with tf.name_scope('softmax'):
            pred = self.softmax_layer(fc)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels))
            tf.summary.scalar("loss", loss)

        with tf.name_scope('regularized_loss'):
            reg_loss = self.regularize(loss, type = self.param.regularize_type, scale=self.param.reg_scale)
            tf.summary.scalar("regularized", reg_loss)
            

        with tf.name_scope('sgd'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(reg_loss, global_step=self.global_step)

        with tf.name_scope('Train_accuracy'):
            acc = tf.equal(tf.argmax(pred, 1), tf.argmax(self.labels, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            tf.summary.scalar("accuracy", acc)

        return pred, optimizer, reg_loss, acc


    def run(self):
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            if self.param.is_training:
                while not coord.should_stop() or sess.should_stop():    
                        #Train
                        _ = self.sess.run(self.optimizer)
                        
            else:
                sum_accurace = 0
                count = 0
            
                while not coord.should_stop() or sess.should_stop():
                        test_accuracy = self.sess.run(self.accuracy)
                        count += 1
                        sum_accurace += test_accuracy* self.param.batch_size
                        if count == self.param.max_steps:
                            break
                print("Test Accuracy: {:.2f}%".format(100.0*(sum_accurace / (self.param.batch_size * count))))
    #                     
        except tf.errors.OutOfRangeError:
            print('Done training...')
        finally:
            coord.request_stop()

        coord.join(threads)
                
                
    
    def prediction(self, video):

        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:     
            while not coord.should_stop() or sess.should_stop():
                    test_accuracy = self.sess.run(self.accuracy)
                    count += 1
                    sum_accurace += test_accuracy* self.param.batch_size
                    if count == self.param.max_steps:
                        break
            print("Test Accuracy: {:.2f}%".format(100.0*(sum_accurace / (self.param.batch_size * count))))
    #                     
        except tf.errors.OutOfRangeError:
            print('Done training...')
        finally:
            coord.request_stop()

        coord.join(threads)

param = paramethers()

param.is_training = True

param.batch_size =  96
param.layer_depth = 4096
param.initial_learning_rate = 5e-3
param.keep_dropout = 0.97
param.epochs = 2000 if param.is_training  else 1
param.max_steps = (param.total_train//param.batch_size)* param.epochs if param.is_training  else (param.total_test//param.batch_size + 1)


inception = inception_cuboid(param)
inception.configure()
inception.run()
                    
