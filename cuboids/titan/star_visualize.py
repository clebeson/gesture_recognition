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
import logging
import glob
import cv2
import scipy.stats as st
import itertools
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import misc
from scipy.misc import imread, imresize
from tensorflow.python.framework import graph_util
from scipy.ndimage.filters import gaussian_filter
#tf.logging.set_verbosity(tf.logging.ERROR)

#import conv_cosnorm as cos
#59% in test

#os.system("pip install pims")

def plot_images(images, subplot = (1,2), show_size=100):
    if not images or len(images) == 0:
        return
    """
    The show_size is the number of pixels to show for each image.
    The max value is 299.
    """
    from skimage.transform import resize
    def normalize_image(x):
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min)
        return x_norm

    # Create figure with sub-plots.

    fig, axes = plt.subplots(*subplot)

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # For each entry in the grid.
    size = len(images)
    for i, ax in enumerate(axes.flat):
        if i >= size: break
        # Get the i'th image and only use the desired pixels.
        img = images[i]
        img = resize(img, (show_size, show_size), anti_aliasing=True)


        # Normalize the image so its pixels are between 0.0 and 1.0
        img_norm = normalize_image(img)

        # Plot the image.
        ax.imshow(img_norm, interpolation=interpolation)

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
def guided_BP(sess, list_images, input_image, logits, labels_id):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        
        
         #define your tensor placeholders for, labels and images
        label_index = tf.placeholder("int64", ())
        label =  tf.one_hot(label_index, logits.shape.as_list()[-1])

        #get the output neuron corresponding to the class of interest (label_id)
        cost = logits * label

        # Guided backpropagtion back to input layer
        gb_grad = tf.gradients(cost, input_image)[0]

        init = tf.global_variables_initializer()

    gbs = []
    for image, label_id in zip(list_images, labels_id):
        output = [0.0]* logits.get_shape().as_list()[1] #one-hot embedding for desired class activations
        output = np.array(output)
        prob = tf.nn.softmax(logits)
        if label_id == -1:
            prob = sess.run(prob, feed_dict={input_image:image, label_index:label_id})
            index = np.argmax(prob)
            print("Predicted_class: ", index)
            output[index] = 1.0

        else:
            output[label_id] = 1.0

        gb_grad_value = sess.run(gb_grad, feed_dict={input_image:image, label_index:label_id})
        gbs.append(gb_grad_value[0])
    return gbs 

def grad_CAM_plus(sess, list_images, input_placeholder, logits,  target_conv_layer, labels_id):
    g = tf.get_default_graph()
   
    #define your tensor placeholders for, labels and images
    label_index = tf.placeholder("int64", ())
    label =  tf.one_hot(label_index, logits.shape.as_list()[-1])

    #get the output neuron corresponding to the class of interest (label_id)
    cost = logits * label

    # Get last convolutional layer gradients for generating gradCAM++ visualization
    target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]
 

    #first_derivative
    first_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad

    #second_derivative
    second_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad 

    #triple_derivative
    triple_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad*target_conv_layer_grad  

    prob = tf.nn.softmax(logits)
    cams = []
    
    for image, label_id in zip(list_images, labels_id):
       
        output = [0.0]*logits.get_shape().as_list()[1] #one-hot embedding for desired class activations
            #creating the output vector for the respective class
        
        output = np.array(output)
        if label_id == -1:
            prob_val = sess.run(prob, feed_dict={input_placeholder: image, label_index:label_id})
            index = np.argmax(prob_val)
            orig_score = prob_val[0][index]
            print("Predicted_class: ", index)
            output[index] = 1.0
            label_id = index
        else:
            output[label_id] = 1.0
        
    
        conv_output, conv_first_grad, conv_second_grad, conv_third_grad = sess.run([target_conv_layer, first_derivative, second_derivative, triple_derivative], feed_dict={input_placeholder:image, label_index:label_id})

        global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        weights = np.maximum(conv_first_grad[0], 0.0)
        #normalizing the alphas
        """	
        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

        alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))
        """

        alphas_thresholding = np.where(weights, alphas, 0.0)

        alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0),axis=0)
        alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))


        alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad[0].shape[2]))



        deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
        #print deep_linearization_weights
        grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0   

        cam = resize(cam,  image.shape[:2])
        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0    
        cam = resize(cam,  image.shape[1:3])
        cams.append(cam)

    
    gbs = guided_BP(sess,list_images, input_placeholder, logits, labels_id )  
    return get_visual_images(list_images, cams, gbs)

def normalize(img, s=0.1):
        '''Normalize the image range for visualization'''
        z = img / np.std(img)
        return np.uint8(np.clip(
            (z - z.mean()) / max(z.std(), 1e-4) * s + 0.5,
            0, 1) * 255)
    
def get_visual_images(imgs, cams, gb_vizes):
    results =  []
    for img, cam, gb_viz in zip(imgs, cams, gb_vizes):
        img = np.squeeze(img)[...,:3]/255.0
        gb_viz = np.dstack((
                gb_viz[:, :, 2],
                gb_viz[:, :, 1],
                gb_viz[:, :, 0],
            ))
        gb_viz_norm = normalize(gb_viz)
        gb_viz -= np.min(gb_viz)
        gb_viz /= gb_viz.max()
        
        gd_img = gb_viz * np.expand_dims(np.minimum(0.25,cam), axis = 2)

        x = np.squeeze(gd_img)
        
        #normalize tensor
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        gd_img = np.clip(x, 0, 255).astype('uint8')

        cam_modified = (cam*-1.0) + 1.0
        cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255*cam_modified), cv2.COLORMAP_JET))
        cam_heatmap = cam_heatmap/255.0
        fin = (img*0.7) + (cam_heatmap*0.3)
        
        fin = (fin*255).astype('uint8')
        img = (img*255).astype('uint8')
        results.append([img, gb_viz_norm, gd_img, cam, cam_heatmap, fin])

    return results

def weights_visualization(sess, weights):
    w = sess.run(weights)
    images = []
    for i in range(w.shape[-1]):
        kernel  = normalize(w[:,:,:,i].squeeze())
        images.append(imresize(kernel, (10,10)))
    return images

       

def filters_visualization(sess, input_placeholder,  filter_name, feature_image = None, feature = 1):
    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-8)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x
    
    tensor1 = tf.get_default_graph().get_tensor_by_name(filter_name)
    tensor2 = tf.get_default_graph().get_tensor_by_name(filter_name.replace("16_1","16_2"))
    tensor3 = tf.get_default_graph().get_tensor_by_name(filter_name.replace("16_1","16_3"))

    #input = tf.get_default_graph().get_tensor_by_name("vgg16_1/import/images:0")
    images = []
    tensor_shape = tensor1.shape.as_list()
    loss = [tf.reduce_mean(tensor1[...,feature]),tf.reduce_mean(tensor2[...,feature]),tf.reduce_mean(tensor3[...,feature])]
    loss =  loss if len( tensor_shape) == 4 else tensor1[0,feature]
    gradient = tf.gradients(loss, input_placeholder)[0]
    

    gradient = tf.nn.l2_normalize(gradient)
    image_shape = input_placeholder.shape.as_list()
    image_shape[0] = 1
    if feature_image is None:
        feature_image = np.random.uniform(size=image_shape)
    
    for i in range(100):
#       feed_dict = {self.dict_model["images"]: feature_image, self.dict_model["labels"]: label, model["keep"]:1.0}
        feed_dict = {input_placeholder: feature_image}
        grad, loss_value = sess.run([gradient, loss],feed_dict=feed_dict)
        grad = np.array(grad).squeeze()
        step_size =  0.1 / (grad.std() + 1e-8)
        feature_image += step_size * grad
        feature_image = gaussian_filter(feature_image, sigma = 0.2)
        
    return deprocess_image(feature_image.squeeze())

def load_inputs(filename, batch_size):
        queue = tf.train.string_input_producer([filename])

        image, label = read_decode_distort(queue)

        image_batch, label_batch = tf.train.batch(
                [image, label], 
                batch_size = batch_size, 
                num_threads = 11, 
                capacity = int(3575* 0.4))

        label_batch = tf.one_hot(label_batch,20)
        return image_batch, label_batch
def read_decode_distort(queue):
        reader = tf.TFRecordReader()
        _, serialized = reader.read(queue)
        features = tf.parse_single_example(serialized, features={
        'video_raw': tf.FixedLenFeature([128 *
                                         128 *
                                         9 ], 
                                        tf.string),
            
        'label': tf.FixedLenFeature([], tf.int64),
        })
        
        video = tf.decode_raw(features['video_raw'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)
        
        video = tf.cast(video, tf.float32)
        video = tf.reshape(video, [128,128,9])
        video = tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(video,0), [124, 120]))
        return video, label



def pbtxt_to_graphdef(filename):
    print("Converting...")
    from google.protobuf import text_format
    with open(filename, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        text_format.Merge(file_content, graph_def)
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, '/notebooks/trained_models/', 'start_vis.pb', as_text=False)



def load_trained_model(images):
           
    with open("/notebooks/trained_models/frozen-graph.pb", mode='rb') as f:
        content = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(content)        
        #graph_def = tf.graph_util.extract_sub_graph(graph_def, ["video_input/batch", "softmax/fc4/matmul-add"])
        tf.import_graph_def(graph_def, input_map={"video_input/batch": images})
            
            
    del content
    graph = tf.get_default_graph()
    
    prob = graph.get_tensor_by_name("import/softmax/fc4/matmul-add:0")
    # net = tf.stop_gradient(net)
    return  prob

def print_graph():
    g = tf.get_default_graph()
    for i in g.get_operations():
        print(i.values()) 
def freeze_graph(model_folder, output_nodes='softmax/fc4/matmul-add', 
                 output_filename='/notebooks/trained_models/frozen-graph.pb', 
                 rename_outputs=None):
    from tensorflow.python.framework import graph_util
    #Load checkpoint 
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    output_graph = output_filename

    #Devices should be cleared to allow Tensorflow to control placement of 
    #graph when loading on different machines
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', 
                                       clear_devices=True)

    graph = tf.get_default_graph()

    onames = output_nodes.split(',')

    #https://stackoverflow.com/a/34399966/4190475
    if rename_outputs is not None:
        nnames = rename_outputs.split(',')
        with graph.as_default():
            for o, n in zip(onames, nnames):
                _out = tf.identity(graph.get_tensor_by_name(o+':0'), name=n)
            onames=nnames

    input_graph_def = graph.as_graph_def()

    # fix batch norm nodes
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, input_checkpoint)

        # In production, graph weights no longer need to be updated
        # graph_util provides utility to change all variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, 
            onames # unrelated nodes will be discarded
        ) 

        # Serialize and write to file
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

def make_prediction(images, prob, labels):
    class_names = ['prendere', 'vieniqui', 'perfetto', 'fame', 'sonostufo', 'seipazzo', 'basta', 'cheduepalle', 'noncenepiu', 'chevuoi',
                'ok', 'combinato', 'freganiente', 'cosatifarei', 'buonissimo', 'vattene', 'messidaccordo', 'daccordo', 'furbo', 'tantotempo']
    ckpt = "/notebooks/cuboids/checkpoints/titan_stars_D3072_pool3_LR-5E-03_drop-0.2_B-96_V-1/"
    with tf.Session() as sess:
       
        # saver.restore(sess, tf.train.latest_checkpoint(ckpt))
        #sess.run(tf.initialize_all_variables())
        #sess.run(init_l)
        print("Steps: ", 3575)
        prediction = {
             "classes": tf.argmax(prob, 1),
             "probs" :  tf.nn.softmax(prob), 
             "labels": tf.argmax(labels, 1)
             }
        predictions = {
            "images" : None,
            "classes":np.array([]),
            "probs": None,
            "labels":np.array([])
        }
        count = 0
        sum_accurace = 0
        
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess, coord=coord)
        target_conv_layer = tf.get_default_graph().get_tensor_by_name("import/flatten/truediv:0")
        try:     
            while not coord.should_stop() or sess.should_stop():
                    img , pred = sess.run([images, prediction])
                    
                    #print(images.shape)
                    if not pred["classes"] == pred["labels"]:
                        #print(img.shape, img.dtype)
                        predictions["images"] = img if predictions["images"] is None else np.concatenate([ predictions["images"], img ] )
                        predictions["classes"] = np.concatenate([ predictions["classes"], pred["classes"] ] )
                        predictions["probs"] = pred["probs"] if  predictions["probs"] is None else np.concatenate([ predictions["probs"], pred["probs"] ], 0)
                        predictions["labels"] = np.concatenate([ predictions["labels"], pred["labels"] ] )
                        
                    count += 1 
                    if count == 3575:
                        break
            accurary = np.mean(predictions["classes"] == predictions["labels"]) 
            print("Test Accuracy: {:.2f}%".format(100.0*accurary))
    #                     
        except tf.errors.OutOfRangeError:
            print('Done test...')
        finally:
            coord.request_stop()

        coord.join(threads)
    
    return predictions



#pbtxt_to_graphdef("/notebooks/trained_models/graph.pbtxt")
tf.reset_default_graph()

name = '/notebooks/video_tfrecord/star_test_rgb.tfrecords'
batch = 100
images, labels = load_inputs(name,batch)

# images = tf.placeholder(tf.float32, [1,  124, 120, 9], name = "input")
# labels = tf.placeholder(tf.float32, [1, 20],name = "labels")
prob = load_trained_model(images)
# print_graph()
#pred = make_prediction(images, prob, labels)

#tf.reset_default_graph()
class_names = ['prendere', 'vieniqui', 'perfetto', 'fame', 'sonostufo', 'seipazzo', 'basta', 'cheduepalle', 'noncenepiu', 'chevuoi',
                'ok', 'combinato', 'freganiente', 'cosatifarei', 'buonissimo', 'vattene', 'messidaccordo', 'daccordo', 'furbo', 'tantotempo']
    
# images = tf.placeholder(tf.float32, [1,  124, 120, 9], name = "input")
# labels = tf.placeholder(tf.float32, [1, 20],name = "labels")
# prob = load_trained_model(images)
# target_conv_layer = tf.get_default_graph().get_tensor_by_name("import/flatten/truediv:0")
        
# with tf.Session() as sess:
#     list_labels = sorted(list(range(20))*len(pred["labels"]))
#     att_maps = grad_CAM_plus(sess, [np.expand_dims(img,0) for img in pred["images"]]*20, images, prob,  target_conv_layer, list_labels )
#     list_img = ["img", "gb_viz_norm", "gd_img", "cam", "cam_heatmap", "fin" ]
#     count = 1
#     for att,l,c, label in zip(att_maps, pred["labels"]*20,pred["classes"]*20, list_labels):
#         name = "{}_{}_{}_{}".format(class_names[int(l)], class_names[label],class_names[int(c)], count)
#         for i, name_img in enumerate(list_img):
#             misc.imsave("/notebooks/attention_maps/{}_{}.jpg".format(name, name_img), att[i])
#         count += 1
#         #plot_images(att, subplot = (1,6), show_size=100)
#     print(count)
prob = tf.get_default_graph().get_tensor_by_name("import/fully_conn/fc1/relu:0")
# prob = tf.get_default_graph().get_tensor_by_name("import/flatten/flatten/Reshape:0")

print("Generate features")
features = np.zeros((3000,61440))
y = np.zeros((3000,))
with tf.Session() as sess:

    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(sess, coord=coord)
    target_conv_layer = tf.get_default_graph().get_tensor_by_name("import/flatten/truediv:0")
    label = tf.argmax(labels,1)
    i =0
    try:     
        while not coord.should_stop() or sess.should_stop():
            feature, l = sess.run([prob, label])
            features[i*batch:(i+1)*batch,:], y[i*batch:(i+1)*batch]= feature, l
            i += 1 
            if i == 3000//batch:
                break
    except tf.errors.OutOfRangeError:
        print('Done test...')
    finally:
        coord.request_stop()

    coord.join(threads)
   

    from sklearn.manifold import TSNE
    tsne = TSNE(n_iter=1000, learning_rate=100.0,metric = 'cosine')
    print("Fit features")
    tsne_results = tsne.fit_transform(features)

    min_1 = tsne_results[:,0].min()
    max_1 = tsne_results[:,0].max()
    min_2 = tsne_results[:,1].min()
    max_2 = tsne_results[:,1].max()
   
    tsne_results[:,0] = (tsne_results[:,0] - min_1) / (max_1 - min_1) 
    tsne_results[:,1] = (tsne_results[:,1] - min_2) / (max_2 - min_2)


    #plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y)
    plt.figure(figsize=(15,13))
    plt.scatter(tsne_results[:, 1], tsne_results[:, 0], c=y,cmap=plt.cm.get_cmap("jet", 20), s=4, edgecolors='none')
    plt.colorbar(ticks=range(20))
    plt.savefig('/notebooks/tsne.png')

    plt.show()


#ckpt = "/notebooks/cuboids/checkpoints/titan_stars_D3072_pool3_LR-5E-03_drop-0.2_B-96_V-1/"
#freeze_graph(ckpt)
#print(images)
#print_graph()

# layers = [ "conv3_1", "conv3_2", "conv3_3"]
# features = [ 256, 256, 256]
# filters = ["Conv2D","BiasAdd", "Relu"]
# with tf.Session() as sess:
#            cd  for f in range(19,20):

#                     image = filters_visualization(sess, images,  "import/softmax/fc4/matmul-add:0",  feature = f)
#                     misc.imsave("/notebooks/star_visual/vgg16_1_softmax_{}.jpg".format(f),image[...,:3])
#                     misc.imsave("/notebooks/star_visual/vgg16_2_softmax_{}.jpg".format(f),image[...,3:6])
#                     misc.imsave("/notebooks/star_visual/vgg16_3_softmax_{}.jpg".format(f),image[...,6:])
#                     print(f)
#                     #plot_images([imageR, imageG, imageB], subplot = (1,3), show_size=100)
    
