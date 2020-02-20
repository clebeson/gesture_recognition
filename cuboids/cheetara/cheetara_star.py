# In[1]:
%matplotlib inline  
from __future__ import division, print_function
import imageio
imageio.plugins.ffmpeg.download()
from random import shuffle as sf
from itertools import izip as zip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pims
import os
import glob
import scipy.stats as st
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import misc
#import conv_cosnorm as cos
#59% in test

#os.system("pip install pims")

class database:
    def __init__(self, database, batch_size):
            self.__path = database
            self.__batch_size = batch_size
            self.__test = -1
            self.__next_batch_id = 0
            self.labels = ["Diving",
                "Golf",
                "Kicking",
                "Lifting",
                "Riding",
                "Run",
                "SkateBoarding",
                "Swing",
                "Walk"]
            
    def __one_hot(self, labels, classes):
        return np.eye(classes)[labels]

    def load(self, depth=40):
        print("Loading dataset...")
        
        files = glob.glob(os.path.join(self.__path, '*.mp4'))
        np.random.shuffle(files)
        self.__dataset = []
        self.__labels = []
        for file in files:
            video, label = self.video_to_images(file)
            self.__dataset.append(video)
            self.__labels.append(label)
        self.__indices = range(len(self.__dataset))
        self.__train_indices = self.__indices
        print(len(self.__dataset)," loaded files.")
        print("Done load...")

    def video_to_images(self, video):
        images = pims.Video(video)
        label = self.get_label(video)
        tensor = np.concatenate(images, 2)
        return tensor, label


    def next_test(self):
        self.__test = self.__test + 1
        if self.__test >= (len(self.__dataset) - 1):
            self.__test = 0
            return None, None

        index = self.__indices[self.__test]
        if self.__test == 0:
            self.__indices.pop(index)
        else:
            self.__indices.pop(index)
            self.__indices.append((self.__test -1))
        test = self.__dataset[self.__test], 
        label = self.__one_hot(self.__labels[self.__test], 9)
        batch = []
        label_batch= []
        
        for _ in range(self.__batch_size):
            batch.append(test)
            label_batch.append(label)

        return  batch, np.squeeze(label_batch)

    def next_batch(self):
        if self.__next_batch_id + self.__batch_size > len(self.__indices):
            self.__next_batch_id = 0
            np.random.shuffle(self.__indices)
        indices = self.__indices[self.__next_batch_id : self.__next_batch_id + self.__batch_size]
        self.__next_batch_id = self.__next_batch_id + self.__batch_size
        
        return  np.squeeze([self.__dataset[i] for i in indices]) , np.squeeze([[self.__one_hot(self.__labels[i], 9)] for i in indices])


    

    # def get_label(self, name):
    #     n, _ = os.path.splitext(name)
    #     n = n.split("/")[-1]
    #     n = n.split("_")[0]
    #     n = int(self.labels.index(n))
    #     return n

    # def get_label(name):
    #     n, _ = os.path.splitext(name)
    #     n = n.split("/")[-1]
    #     n = n.split("_")[-1]
    #     n = int(n)-1
    #     return n
    def get_label(name):
        n, _ = os.path.splitext(name)
        n = int(n.split("-")[-1])
        return n

    def video_to_tensor(self, video, h=90, w=80, depth = 40):
            num = 0
            label= None
            
    #     try:
        
            images = pims.Video(video)
            num = len(images)
            ratio = float(num) / depth if num > depth else 1.0
            if num <= 0:
                return [], None
            im_shape = images[0].shape
            black = np.zeros((h,w))
            tensor = np.ndarray((h,w,depth))
            # img_prev = images[0]
            # img_prev = (0.2989 * img_prev[0:h,0:w,0] + 0.5870 * img_prev[0:h,0:w,1] + 0.1140 * img_prev[0:h,0:w,2])
        
            for i in xrange(1,depth):
                index = int(round(ratio*i))
                if(index<num):
                    img_color = images[index]
                    img_gray = (0.2989 * img_color[:,:,0] + 0.5870 * img_color[:,:,1] + 0.1140 * img_color[:,:,2])
                    
    #                 image = np.abs(img_prev - img_gray)
                    
                    image = misc.imresize(img_gray, (h,w))
            
                else:
    #                 print("black", index, num)
                    image = black

                        
                    
    #                 image  = np.transpose(img, (2, 0, 1))
                
                tensor[:,:,i]=image
            
            del images
            
            label = self.get_label(video)
        
            return tensor.astype(np.float32), label

class paramethers:
    def __init__(self):
        self.is_training = True
        
        #Input parameters
        self.batch_size = 96
        self.rows, self.cols, self.channels, self.depth= 128, 128, 9, 40
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
        self.summary_step = 1000
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
        self.vggpool=4
        self.is_tfrecord = True
        self.tfrecord_base_name="last_{}_sampled_pad".format(self.depth)
        self.tfrecord_base_name_test="last_{}_sampled_pad".format(self.depth)
        self.vgg_layer_name = "import/pool{}:0".format(self.vggpool)
        self.save_checkpoint_secs = 600
        self.regularize_type = 2
        self.cuboids_size = 20
        self.database = "../dataset/train_split"
        self.is_monitored = True
        self.version = 1
        
class inception_cuboid:
    def __init__(self, param = None):
        
        self.param = paramethers() if param is None else param
        self.sess = None

    def __create_graph(self):
        print("Resetting Graph...")
        if (self.sess is not None):
            self.sess.close()

        tf.reset_default_graph()
        tf.Graph().as_default()
        tf.set_random_seed(2018)
        
    def __create_model_and_session(self, is_monitored = True):
        print("Creating Session...")
        #print("Batch size: ",self.param.batch_size)
        #print("tensorboard --logdir={}\n\n".format(self.checkpoint_dir))
        self.model = self.star_vgg_model()
       
        iter_per_epoch = self.param.max_steps//self.param.epochs
        #print("iter per epoch:",iter_per_epoch)
        if self.param.is_monitored: 
            def formatter(d): 
                return ' epoch {:d}, loss: {:.2f},  train_acc: {:.2f}%'.format(d.values()[2]//iter_per_epoch, d.values()[0], d.values()[1] * 100)

            # with tf.device('/gpu:0'):
            hooks = [
                tf.train.StopAtStepHook(num_steps=self.param.max_steps),
                tf.train.LoggingTensorHook([self.model["loss"], self.model["accuracy"], self.global_step], every_n_iter=iter_per_epoch, formatter=formatter),
            #     tf.train.NanTensorHook(loss),
            ]

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = not self.param.is_training
            config.intra_op_parallelism_threads = 44
            config.inter_op_parallelism_threads = 44
            
            self.sess = tf.train.MonitoredTrainingSession(checkpoint_dir=self.checkpoint_dir,
                                                hooks=hooks,
                                                save_checkpoint_secs=self.param.save_checkpoint_secs,
                                                save_summaries_steps=self.param.summary_step,
                                                config=config)
        else:
           self.sess = tf.Session() 
           init = tf.global_variables_initializer()
           self.sess.run(init)    
       
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
                                            save_summaries_steps=self.param.summary_step)

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

    def __configure(self):
        self.__create_graph()

        self.keep = tf.placeholder(tf.float32)
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(self.param.initial_learning_rate, self.global_step, 
                                                self.param.decay_steps, self.param.decay_rate, staircase=True)

        tf.summary.scalar("learning_rate", self.learning_rate)
        self.tfrecord_input_file = '/notebooks/video_tfrecord/{}.tfrecords'.format( "star_train_rgb_aug_128" if self.param.is_training  else "star_test_rgb_aug_128")
        self.checkpoint_dir="/notebooks/cuboids/checkpoints/cheetara_stars_D{}_pool{}_LR-{:.0E}_drop-{}_B-{}_V-{}".format( self.param.layer_depth,
                                                                                                                    self.param.vggpool, 
                                                                                                                    self.param.initial_learning_rate, 
                                                                                                                    self.param.keep_dropout, 
                                                                                                                    self.param.batch_size,
                                                                                                                    self.param.version )
                                                                                    
        if not self.param.is_training:
            self.param.batch_size = 5 
            self.param.max_steps = self.param.total_test//self.param.batch_size + 1
           
                                                                                                               
        if self.param.is_tfrecord:
            with tf.device('/cpu:0'):
                with tf.name_scope('video_input'):
                    self.input_tensor, self.labels = self.__load_inputs(self.tfrecord_input_file, self.param.batch_size)
                    # dataset = tf.data.TFRecordDataset(self.tfrecord_input_file)
                    # dataset  = dataset.map(self._parse_function)
                    # dataset = dataset.shuffle(buffer_size=self.param.total_train)
                    # dataset = dataset.repeat(self.param.epochs)
                    # dataset = dataset.batch(self.param.batch_size)

                    # print(self.param.batch_size,dataset)

                    # iterator = tf.data.Iterator.from_structure(dataset.output_types,
                    #                        dataset.output_shapes)

                    # self.next_element =  iterator.get_next()
                    # self.training_init_op = iterator.make_initializer(dataset)

                    # self.input_tensor = tf.placeholder(tf.float32, [self.param.batch_size,  124, 120, self.param.channels], name = "video_input")
                    # self.labels = tf.placeholder(tf.int32, [self.param.batch_size, self.param.n_classes],name = "labels")
                    

                    # self.iterator = dataset.make_initializable_iterator()





            print("input file:", self.tfrecord_input_file)
            print("Check point dir: ", self.checkpoint_dir)
        else:
            self.R = tf.placeholder(tf.float32, [self.param.batch_size,  self.param.rows-4, self.param.cols-10, self.param.channels], name = "video_inputR")
            self.G = tf.placeholder(tf.float32, [self.param.batch_size,  self.param.rows-4, self.param.cols-10, self.param.channels], name = "video_inputG")
            self.B = tf.placeholder(tf.float32, [self.param.batch_size,  self.param.rows-4, self.param.cols-10, self.param.channels], name = "video_inputB")
            self.labels = tf.placeholder(tf.float32, [self.param.batch_size, self.param.n_classes],name = "labels")

        
        #self.__create_inseption_model()
        self.__create_model_and_session()
        
    def squeeze(self,tensor):
        shape = tensor.shape.as_list()
        first = shape[0]
        if 1 in shape[1:]:
            tensor = tf.squeeze(tensor)
            if first == 1:
                tensor = tf.expand_dims(tensor,0)
        return tensor
            
    def fc_cosnorm(self,x, w, bias=0.00001):
        x = tf.add(x, bias)
        w = tf.add(w, bias)

        #   x_shape = tf.shape(x)
        #   x_b = tf.fill([x_shape[0], 1], bias)
        #   x = tf.concat( [x_b, x], 1)

        #   w_shape = tf.shape(w)
        #   w_b = tf.fill([1, w_shape[1]], bias)
        #   w = tf.concat([w_b , w],0)
        y = tf.matmul(x, w)

        x = tf.reduce_sum(tf.square(x),1, keepdims=True)
        x = tf.sqrt(x)
            
        w = tf.reduce_sum(tf.square(w),0, keepdims=True)
        w = tf.sqrt(w)

       
        return y / (x * w)

    def conv3d(self, tensor, temp_kernel, space_kernel, num_filters, stride=[1,1,1,1,1], name = "conv3d"):   

        channels = int(tensor.shape[4])
        filter, _ = self.get_3d_filters(temp_kernel, channels, num_filters, space_kernel, name)
       
        temp_layer = tf.nn.conv3d(tensor, filter, stride, data_format= "NDHWC", padding='VALID', name=name)
        # temp_layer = cos.conv3d_cosnorm(input, filters, strides=[1,1,1,1,1], padding='VALID')
        temp_layer =  self.batch_norm(temp_layer)
        #temp_layer = tf.nn.bias_add(temp_layer, bias)
        
        
        temp_layer = tf.nn.relu(temp_layer)
        return temp_layer

    def get_3d_filters(self, temporal_kernel_size, channels, temp_filters_size, spacial_kernel_size=1, id = 0):
            filter = tf.Variable(tf.random_normal([ temporal_kernel_size, spacial_kernel_size, spacial_kernel_size,   channels, temp_filters_size ]), 
                            dtype=tf.float32, name="3d_filter")
            bias = tf.Variable(tf.random_normal([temp_filters_size]), name="B_temp")
            return filter, bias
    
    def get_fc_weights(self, w_inputs, w_output, id=0):
        
        weight= tf.Variable(tf.truncated_normal([w_inputs, w_output]), name="{}/weight".format(id))
        bias =  tf.Variable(tf.truncated_normal([w_output]), name="{}/bias".format(id))
        return weight, bias

    def fully_conn(self, previous_layer, in_size, out_size, keep, layer_id):
        #print("drop out:", keep)
        w, b = self.get_fc_weights(in_size, out_size, layer_id)
        # fc_layer = tf.matmul(previous_layer, w)
        # fc_layer = self.batch_norm(fc_layer, scope = "BN_fc/{}".format(layer_id))
        #fc_layer = tf.nn.bias_add(fc_layer, b, name="{}/matmul-add".format(layer_id))
        fc_layer = tf.nn.bias_add(self.fc_cosnorm(previous_layer, w), b, name="{}/matmul-add".format(layer_id))
        fc_layer = tf.nn.dropout(fc_layer, keep, name="{}/dropout".format(layer_id))
        fc_layer = tf.nn.relu(fc_layer, name="{}/relu".format(layer_id))
        return fc_layer

    def avgpool(self, tensor, k=2, d=1):
        if len(tensor.shape) == 5:
            return tf.nn.avg_pool3d(tensor, ksize=[1,  d,k, k, 1], strides=[1, d , k, k, 1], data_format= "NDHWC" , padding='VALID')
        
        return tf.nn.avg_pool(tensor, ksize=[1, k, k, d], strides=[1, k, k, d], padding='VALID' )
                
    def maxpool(self, tensor, k=2, d=1):
        if len(tensor.shape) == 5:
            return tf.nn.max_pool3d(tensor, ksize=[1,  d,k, k, 1], strides=[1, d , k, k, 1], data_format= "NDHWC" , padding='VALID')
        
        return tf.nn.max_pool(tensor, ksize=[1, k, k, d], strides=[1, k, k, d], padding='VALID' )
                                                    
    def tensor_maxpool(self, tensor, axis = 4, layer_id = 0):
        if len(tensor.shape) == 4:
            tensor=tf.expand_dims(tensor,1)
        tensor_mp = tf.reduce_max(tensor, axis=axis, keepdims = True)
        return tensor_mp

    def tensor_sum(self, tensor, axis = 4, layer_id = 0):
        tensor_sum = tf.reduce_sum(tensor, axis=axis, keepdims = True)
        # tensor_mp =  batch_norm(tensor_mp, scope = "BN_temp/{}".format(layer_id))
        return tensor_sum

    def tensor_avgpool(self, tensor, axis = 4, layer_id = 0):
        tensor_ap = tf.reduce_mean(tensor, axis=axis)
        
        # tensor_ap =  batch_norm(tensor_ap, scope = "BN_space/{}".format(layer_id))
        return tensor_ap

    def change_format(self, tensor):
        if len(tensor.shape) == 4:
            tensor=tf.expand_dims(tensor,-1)
            tensor = tf.transpose(tensor, perm=[0, 3, 1,2,4])
        #print("change format ",tensor)
        return tensor

    def cuboid_model(self, tensor, kernel_size,  num_filters, id, spac_kernel=3):

        input = self.change_format(tensor)
        smoothed = input
        if spac_kernel>1:
            smoothed = self.conv3d(smoothed, 1, spac_kernel, 1, [1,1,1,1,1], name="smooth")
            #print("smoothed:", smoothed)
            smoothed =  self.change_format(smoothed)

        
        #Temporal convolution (such as quadrature pair filters)
        # quadracture_pair  = self.temporal_conv(smoothed, kernel_size, channels, num_filters, layer_id = id)
        temporal  = self.conv3d(smoothed, kernel_size, 1, num_filters, [1,2,1,1,1], name="temporal")
        #print("temporal", temporal)
        cuboid_layer  = self.conv3d(temporal, 1, 1, 1, [1,1,1,1,1], name="reduce_temporal")
        #Sum of the quadrature pair
        #cuboid_layer = self.tensor_maxpool(temporal, layer_id=id)
        # cuboid_layer = self.squeeze(tf.transpose(temporal, perm=[0, 4, 2,3,1]))
        #print("cuboid", cuboid_layer )
        # cuboid_layer =  rgb_representation(cuboid_layer);
        return cuboid_layer

    def star_vgg_model(self):
        with tf.device('/cpu:0'):
            input = self.squeeze(self.input_tensor)
            # input = self.batch_norm(input, "BN_input")

        if self.param.is_tfrecord:
            self.R = tf.slice(input,[0,0,0,0],[self.param.batch_size,self.param.rows-4, self.param.cols-10,3])
            self.G = tf.slice(input,[0,0,0,3],[self.param.batch_size,self.param.rows-4, self.param.cols-10,3])
            self.B = tf.slice(input,[0,0,0,6],[self.param.batch_size,self.param.rows-4, self.param.cols-10,3])
        
            
        # tf.summary.image("images_cuboids", rgb)
        with tf.name_scope('vgg16_1'):
             vgg1,_ = self.__load_trained_model(self.R, 'vgg16_1')
        with tf.name_scope('vgg16_2'):
             vgg2,_ = self.__load_trained_model(self.G, 'vgg16_2')
        with tf.name_scope('vgg16_3'):
             vgg3,_ = self.__load_trained_model(self.B, 'vgg16_3')
             #print("VGG output :", vgg)
        with tf.name_scope('flatten'):
            ensemble = (vgg1+vgg2+vgg3)/3.0
            flatten = tf.layers.flatten(ensemble)
            #print("flatten:", flatten)


        
        with tf.name_scope('fully_conn'):
            fc = self.fc_model(flatten, keep = self.param.keep_dropout)
            
        with tf.name_scope('softmax'):
            logits = self.softmax_layer(fc)
            predictions = {
             "classes": tf.argmax(logits, 1),
             "probs" :  tf.nn.softmax(logits), 
             "labels": tf.argmax(self.labels, 1)
             }

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.labels))
            tf.summary.scalar("loss", loss)

        with tf.name_scope('regularized_loss'):
            reg_loss = self.regularize(loss, type = self.param.regularize_type, scale=self.param.reg_scale)
            tf.summary.scalar("regularized", reg_loss)
            

        with tf.name_scope('sgd'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(reg_loss, global_step=self.global_step)

        with tf.name_scope('Train_accuracy'):
            acc = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            tf.summary.scalar("accuracy", acc)
       
        model = {
            "logits" : logits,
            "loss" : reg_loss,
            "optimizer": optimizer,
            "accuracy": acc,
            "predictions":predictions
        }

        return model
        

        # output = rgb_representation(self.squeeze(output));
        return output

    def cuboid_vgg_model(self):
        input = self.squeeze(self.input_tensor)
        input = self.batch_norm(input, "BN_input")
        #print(input)
        
        with tf.name_scope('cuboid_1'):
            cuboid= self.cuboid_model(input, 7, self.param.cuboids_size, 1, 7)

        with tf.name_scope('cuboid_2'):
            cuboid= self.cuboid_model(cuboid, 5, self.param.cuboids_size, 2, 5)

        with tf.name_scope('cuboid_3'):
            cuboid= self.cuboid_model(cuboid, 2, self.param.cuboids_size, 1, 3)
            #cuboid = self.maxpool(cuboid)

        with tf.name_scope('Avg_Pool'):
            cuboid = self.avgpool(cuboid)
            #cuboid = self.batch_norm(cuboid, "BN_cubod_1_3")
       
        #print("RGB:",cuboid)
     
        rgb = tf.transpose(self.squeeze(cuboid), perm=[0,2,3,1])
        tf.summary.image("images_cuboids", rgb)
        with tf.name_scope('vgg16'):
             vgg,_ = self.__load_trained_model(rgb, 'vgg16')
             #print("VGG output :", vgg)
        with tf.name_scope('flatten'):
            flatten = tf.layers.flatten(vgg)
            #print("flatten:", flatten)


        
        with tf.name_scope('fully_conn'):
            fc = self.fc_model(flatten, keep = self.keep)
            
        with tf.name_scope('softmax'):
            logits = self.softmax_layer(fc)
            predictions = {
             "classes": tf.argmax(logits, 1),
             "probs" :  tf.nn.softmax(logits), 
             "labels": tf.argmax(self.labels, 1)
             }

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
            tf.summary.scalar("loss", loss)

        with tf.name_scope('regularized_loss'):
            reg_loss = self.regularize(loss, type = self.param.regularize_type, scale=self.param.reg_scale)
            tf.summary.scalar("regularized", reg_loss)
            

        with tf.name_scope('sgd'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(reg_loss, global_step=self.global_step)

        with tf.name_scope('Train_accuracy'):
            acc = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            tf.summary.scalar("accuracy", acc)
       
        model = {
            "logits" : logits,
            "loss" : reg_loss,
            "optimizer": optimizer,
            "accuracy": acc,
            "predictions":predictions
        }

        return model
        

        # output = rgb_representation(self.squeeze(output));
        return output

    def inseptinon_layer(self, tensor, channels, kernels_size, id):
        
        c1= tf.expand_dims(self.cuboid_model(tensor, kernels_size[0], channels, 2, "{}_{}".format(id,1)), -1)
        c2= tf.expand_dims(self.cuboid_model(tensor, kernels_size[1], channels, 2, "{}_{}".format(id,2)), -1)
        c3= tf.expand_dims(self.cuboid_model(tensor, kernels_size[2], channels, 2, "{}_{}".format(id,3)), -1)
        stack=self.squeeze(tf.stack([c1,c2,c3],4))
        output = self.tensor_avgpool(stack, layer_id=id)
        print("inseption out:", output)

        # output = rgb_representation(self.squeeze(output));
        return output

    def conv(self,x, kernel_size):
        depth = x.shape.as_list()[-1]
        W, b = self.get_conv2d_filters(kernel_size, depth)
        x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],data_format= "NHWC", padding='VALID' )
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
        return x 
        
    def batch_norm(self, tensor, scope = None):
        #return tensor
        return tf.layers.batch_normalization(tensor, name = scope, training = self.param.is_training )

    def rgb_representation(self, tensor):
        max = tf.reduce_max(tensor, axis=3, keepdims = True)
        #m, v = tf.nn.moments(tensor, [3], keepdims = True)
        # rgb = tf.stack([max, m, v], axis = 3)
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
        'video_raw': tf.FixedLenFeature([self.param.rows *
                                         self.param.cols *
                                         self.param.channels ], 
                                        tf.string),
            
        'label': tf.FixedLenFeature([], tf.int64),
        })
        
        video = tf.decode_raw(features['video_raw'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)
        
        video = tf.cast(video, tf.float32)
        video = tf.reshape(video, [self.param.rows, self.param.cols, self.param.channels])
        if self.param.is_training:
            
            video = tf.random_crop(video, [self.param.rows-4, self.param.cols-10, self.param.channels])
            video = tf.image.random_flip_left_right(video)

        else:
            video = tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(video,0), [self.param.rows-4, self.param.cols-10]))
        return video, label

    def __load_inputs(self, filename, batch_size):
        queue = tf.train.string_input_producer([filename])

        image, label = self.__read_decode_distort(queue)
        if self.param.is_training:
            image_batch, label_batch = tf.train.shuffle_batch(
                [image, label], 
                batch_size = self.param.batch_size, 
                num_threads =11, 
                capacity = int(self.param.total_train * 0.05) + 3 * self.param.batch_size, 
                min_after_dequeue = int(self.param.total_train * 0.05))
        else:
            image_batch, label_batch = tf.train.batch(
                [image, label], 
                batch_size = self.param.batch_size, 
                num_threads = 11, 
                capacity = int(self.param.total_test * 0.4))

        label_batch = tf.one_hot(label_batch, self.param.n_classes)
        return image_batch, label_batch

    def _parse_function(self, example_proto):
        features = tf.parse_single_example(example_proto, features={
        'image_raw': tf.FixedLenFeature([self.param.rows *
                                         self.param.cols *
                                         self.param.channels ], 
                                        tf.string),
            
        'label': tf.FixedLenFeature([], tf.int64),
        })
        
        video = tf.decode_raw(features['image_raw'], tf.uint8)
        video = tf.cast(video, tf.float32)
        video = tf.reshape(video, [self.param.rows, self.param.cols, self.param.channels])
        if self.param.is_training:  
            video = tf.random_crop(video, [self.param.rows-4, self.param.cols-10, self.param.channels])
            video = tf.image.random_flip_left_right(video)
        
        else:
            video = tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(video,0), [self.param.rows-4, self.param.cols-10]))

        label = features['label']
        label = tf.cast(tf.one_hot(label, self.param.n_classes), tf.int32)
        
        
        return video, label

        # Creates a dataset that reads all of the examples from two files, and extracts
        # the image and label features.
       
    def fc_model(self, input, keep):
        out_shape = input.shape.as_list()
        keep = keep if  param.is_training else 1.0
        print("Keep :",keep)
        
        net = self.fully_conn(input, np.prod(out_shape[1:]), self.param.layer_depth,keep,  "fc1" )
        # net = self.fully_conn(net, self.param.layer_depth , self.param.layer_depth, keep,"fc2" )
        #net = self.fully_conn(net, self.param.layer_depth, self.param.layer_depth,  keep,"fc3" )
        return net

    def softmax_layer(self, fc_layer):
        out_shape = fc_layer.shape.as_list()
        w, b = self.get_fc_weights(np.prod(out_shape[1:]), self.param.n_classes, "softmax")
        logits = tf.add(tf.matmul(fc_layer, w), b, name="fc4/matmul-add")
        return logits

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
        # net = tf.stop_gradient(net)
        return net, graph
        
    def inception_model(self):
        input = self.squeeze(self.input_tensor)
        input = self.batch_norm(input, "BN_input")
        print(input)
            
        # with tf.name_scope('spacial'):
        #     spacial = self.squeeze(self.conv_tensor(input, 5))
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
   
    def train(self):
        
        self.__configure()
        print("tensorboard --logdir={}\n\n".format(self.checkpoint_dir))
    
        if self.param.is_tfrecord:
            # self.sess.run(self.training_init_op)
            
            sum = 0
            count = 0
            step_avg = 300
            coord = tf.train.Coordinator() 
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            try:
                   
                    while not coord.should_stop() or sess.should_stop():    
                        #Train
                        # data, labels  = self.sess.run(next_element)
                        _, acc = self.sess.run([self.model["optimizer"], self.model["accuracy"]])    #, feed_dict={self.input_tensor:data, self.labels:labels}
                        
                        
                        if count == step_avg:
                            if (sum/step_avg) > 0.99:
                                break
                            sum = 0 
                            count = 0
                        sum = sum + acc
                        count = count +1
                            
                    
            except tf.errors.OutOfRangeError:
                print('Done training... An error was ocurred during the training!!!')
            finally:
                coord.request_stop()

            coord.join(threads)
            print('Done training...')
        else:
            load_database(self.param.database)
            step = 0
            while(step < self.param.max_steps):
                batch, labels = get_next_batch();      
                _, step = self.sess.run([self.model["optimizer"], self.global_step], feed_dict={"input_tensor":batch, "labels":labels})
       # return statistics_list
     
    def train_cross_validation(self, dataset):
        os.system("rm -rf /notebooks/cuboids/checkpoints/*cross*")
        self.param.is_tfrecord = False
        self.param.is_monitored = True
        data = database(dataset, self.param.batch_size)
        data.load(self.param.depth)

        iter_per_epoch = self.param.max_steps // self.param.epochs
        
        test_data, label = data.next_test()
        test_data = np.expand_dims(np.squeeze(test_data),1)

        errors = 0
        count_fold = 0

        while test_data is not None:
            print("\n-----------------------------------------------------")
            print("Fold: ",count_fold, " (cumulative Error: {})".format(errors))
            self.param.keep_dropout = count_fold
            self.__configure()
            got = False
            step = 0
            print("Training...")
            while( step < (self.param.max_steps-1) and not got):
                batch, labels = data.next_batch()

                batch = np.expand_dims(batch,1)
                
                _,step= self.sess.run([self.model["optimizer"], self.global_step], feed_dict={self.input_tensor:batch, self.labels:labels, self.keep:1.0})
                step = step+1
                #epoch = step / iter_per_epoch
                # if epoch % 50 == 0:
                #     accuracy=self.sess.run(self.model["accuracy"], feed_dict={self.input_tensor:batch, self.labels:labels})
                #     print("{:>11}: {} - Accuracy: {:.2f}%".format("-> Epoch",epoch, (accuracy*100) ))  
                
                if  step % iter_per_epoch == 1:
                   
                    acc = self.sess.run(self.model["accuracy"], feed_dict={self.R:test_data[:,:,:3], self.G:test_data[:,:,3:6],self.B:test_data[:,:,6:9], self.labels:label, self.keep:1.0})
                   
                    if acc >  0.98: 
                        got = True
                        print("Accuracy Test : ",acc)

            if not got:
                errors = errors + 1
            test_data, label = data.next_test()
            test_data = np.expand_dims(np.squeeze(test_data),1)
            count_fold = count_fold + 1
        print("Cross Accuracy:{:.2f}%".format((errors*100/count_fold)))

       # return statistics_list
                
    def generate_confusion_matrix(self, statistics):
        print("Amount: ",len(statistics["labels"]))
        def plot_confusion_matrix(cm, classes,
                                    normalize=False,
                                    title='Confusion matrix',
                                    cmap=plt.cm.Blues):
                """
                This function prints and plots the confusion matrix.
                Normalization can be applied by setting `normalize=True`.
                """
                if normalize:
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    print("Normalized confusion matrix")
                else:
                    print('Confusion matrix, without normalization')

                print(cm)

                plt.imshow(cm, interpolation='nearest', cmap=cmap)
                plt.title(title)
                plt.colorbar()
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)

                fmt = '.2f' if normalize else 'd'
                thresh = cm.max() / 2.
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, i, format(cm[i, j], fmt),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('Real')
                plt.xlabel('Predito')
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(statistics["labels"],statistics["classes"])
        np.set_printoptions(precision=2)
        # class_names = ['prendere', 'vieniqui', 'perfetto', 'fame', 'sonostufo', 'seipazzo', 'basta', 'cheduepalle', 'noncenepiu', 'chevuoi',
        #         'ok', 'combinato', 'freganiente', 'cosatifarei', 'buonissimo', 'vattene', 'messidaccordo', 'daccordo', 'furbo', 'tantotempo']

        class_names = ["abort", "circle", "hello", "no", "stop", "turn", "turn_left", "turn_right", "warn"]

        # Plot non-normalized confusion matrix
        plt.figure(figsize=(15,15))
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                            title='Confusion matrix, without normalization')

        # # Plot normalized confusion matrix
        plt.figure(figsize=(15,15))
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')

        
        #plt.savefig("/notebooks/cuboids/confusion_matrix.png")
        plt.show()

    def make_prediction(self, videos = None):
        self.__configure()
        print("Steps: ", self.param.max_steps)
        predictions = {"classes":[],
                      "probs":[],
                      "labels":[]
        }
        count = 0
        sum_accurace = 0
        if self.param.is_tfrecord:
            coord = tf.train.Coordinator() 
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            try:     
                while not coord.should_stop() or sess.should_stop():
                        test_accuracy, pred = self.sess.run([self.model["accuracy"], self.model["predictions"]])
                        predictions["classes"].extend(pred["classes"])
                        predictions["probs"].extend(pred["probs"])
                        predictions["labels"].extend(pred["labels"])
                        count += 1
                        sum_accurace += test_accuracy* self.param.batch_size
                        if count == self.param.max_steps:
                            break
                print("Test Accuracy: {:.2f}%".format(100.0*(sum_accurace / (self.param.batch_size * count))))
        #                     
            except tf.errors.OutOfRangeError:
                print('Done test...')
            finally:
                coord.request_stop()

            coord.join(threads)
        elif videos is not None:
            predictions = {"classes":[],
                      "probs":[] 
                      }
            
            for video in videos:
                pred = self.sess.run(self.model["predictions"],feed_dict={"input_tensor":video})
                predictions["classes"].extend(pred["classes"])
                predictions["probs"].extend(pred["probs"])
        print("Prediction is done")
        return predictions


#85.09
#stars_train_D1024_FTFS-8_FS-2_SFS-64_TKS-5_KS-6_pool3_LR-5E-03_drop-0.95_B64

param = paramethers()

param.is_training = True
param.batch_size =  96
param.layer_depth = 3072
param.channels = 9
param.initial_learning_rate = 5e-3
param.reg_scale = 0.0001
param.regularize_type = 2
param.vggpool = 3
param.vgg_layer_name = "import/pool{}:0".format(param.vggpool)
param.total_train = 68470
param.total_test = 3579
param.keep_dropout = 0.2
param.version = 1
param.rows, param.cols, param.channels, param.depth= 96, 128, 9, 40


param.epochs = 3000 if param.is_training  else 1
param.max_steps = (param.total_train//param.batch_size)* param.epochs if param.is_training  else (param.total_test//param.batch_size + 1)
param.decay_steps = (1* param.total_train// param.batch_size) + 1
param.decay_rate = 0.98
param.save_checkpoint_secs = 600


# param.batch_size =  96
# param.layer_depth = 4096
# param.initial_learning_rate = 5e-3
# param.reg_scale = 1e-4
# param.regularize_type = 2
# param.vggpool = 4
# param.keep_dropout = 0.79 #81
# param.epochs = 4000 if param.is_training  else 1
# param.max_steps = (param.total_train//param.batch_size)* param.epochs if param.is_training  else (param.total_test//param.batch_size + 1)
# param.decay_steps = param.total_train// param.batch_size + 1
# param.decay_rate = 0.975
# param.save_checkpoint_secs = 300


inception = inception_cuboid(param)
# inception.configure()
if(param.is_training):
    inception.train()
    #inception.train_cross_validation("/notebooks/datasets/videos_star")
else:
    statistics = inception.make_prediction()
    inception.generate_confusion_matrix(statistics)
  



# def generate_confusion_matrix(statistics):
#         print("Amount: ",len(statistics["labels"]))
#         def plot_confusion_matrix(cm, classes,
#                                     normalize=False,
#                                     title='Confusion matrix',
#                                     cmap=plt.cm.Blues):
#                 """
#                 This function prints and plots the confusion matrix.
#                 Normalization can be applied by setting `normalize=True`.
#                 """
#                 if normalize:
#                     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
#                     print("Normalized confusion matrix")
#                 else:
#                     print('Matriz de Confusão')

#                 # print(cm)

#                 plt.imshow(cm, interpolation='nearest', cmap=cmap)
#                 plt.title(title)
#                 plt.colorbar()
#                 tick_marks = np.arange(len(classes))
#                 plt.xticks(tick_marks, classes, rotation=45)
#                 plt.yticks(tick_marks, classes)

#                 fmt = '.2f' if normalize else 'd'
#                 thresh = cm.max() / 2.
#                 for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    
#                     plt.text(j, i, format(cm[i, j], fmt), 
#                             fontsize=12,
#                             horizontalalignment="center",
#                             color="white" if cm[i, j] > thresh else "black")

#                 plt.tight_layout()
#                 plt.ylabel('Real')
#                 plt.xlabel('Predito')
#         # Compute confusion matrix
#         cnf_matrix = confusion_matrix(statistics["labels"],statistics["classes"])
#         np.set_printoptions(precision=2)
#         class_names = ['prendere', 'vieniqui', 'perfetto', 'fame', 'sonostufo', 'seipazzo', 'basta', 'cheduepalle', 'noncenepiu', 'chevuoi',
#                 'ok', 'combinato', 'freganiente', 'cosatifarei', 'buonissimo', 'vattene', 'messidaccordo', 'daccordo', 'furbo', 'tantotempo']

#         # # Plot non-normalized confusion matrix
#         # plt.figure(figsize=(10,10))
#         # plot_confusion_matrix(cnf_matrix, classes=class_names,
#         #                     title='Confusion matrix, without normalization')

#         # # Plot normalized confusion matrix
#         plt.figure(figsize=(13,10))
#         plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                             title='Normalized confusion matrix')

        
#         plt.savefig("/notebooks/cuboids/confusion_matrix_90_93.png")
#         plt.show()
# generate_confusion_matrix(statistics)

# import os
# file = "/notebooks/cuboids/checkpoints/stars_D3072_pool3_LR-5E-03_drop-0.3_B-96_V-1"
# os.system("rm -r {}".format(file))


