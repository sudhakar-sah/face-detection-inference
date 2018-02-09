""" 
This layer accepts input as the output of model (encoded y_pred) and 
returns detections in proper format after decoding and then applying nms 
Purpose of this layer to make it easier to port in mobile which is being done using JNI interface currently
"""



import keras.backend as K 
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer 
import numpy as np 
import tensorflow as tf

# def decode_y2_tf(y_pred, 
#                   confidence_thresh=0.5,
#                   iou_threshold=0.45,
#                   top_k='all',
#                   input_coords='centroids',
#                   normalize_coords=False,
#                   img_height=None,
#                   img_width=None):


#     #y_pred_tensor = tf.stack(y_pred)
#     y_pred_tensor = y_pred

#     #print (sess.run(y_pred_tensor))
#     y_pred_converted_tensor = y_pred_tensor[:,:,-14:-8]

#     conf, idx = tf.nn.top_k(y_pred_tensor[:,:,:-12])
#     class_idx = tf.cast(idx, tf.float32)


#     pred_4_1 = tf.exp(tf.multiply(y_pred_converted_tensor[:,:,4], y_pred_tensor[:,:,-2]))
#     pred_5_1 = tf.exp(tf.multiply(y_pred_converted_tensor[:,:,5], y_pred_tensor[:,:,-1]))
#     pred_4_2 = tf.add(pred_4_1,(tf.multiply(pred_4_1, y_pred_tensor[:,:,-6])))
#     pred_5_2 = tf.add(pred_5_1, (tf.multiply(pred_5_1, y_pred_tensor[:,:,-5])))

#     pred_2_2 = tf.add(y_pred[:,:,-8], tf.multiply(y_pred_converted_tensor[:,:,2], tf.multiply(y_pred_tensor[:,:,-4], y_pred_tensor[:,:,-6])))

#     pred_3_2 = tf.add(y_pred[:,:,-7], tf.multiply(y_pred_converted_tensor[:,:,3], tf.multiply(y_pred_tensor[:,:,-3], y_pred_tensor[:,:,-5])))
    
    # cx =  pred_2_2
    # cy =  pred_3_2
    # w = pred_4_2
    # h = pred_5_2

#     # xmin = tf.subtract(cx, tf.div(w, 2.))
#     # xmax = tf.add(cx, tf.div(w, 2.))
#     # ymin = tf.subtract(cy, tf.div(h, 2.))
#     # ymax = tf.add(cy, tf.div(h, 2.))

    # xmin = tf.maximum(cx - w/2.,0)
    # xmax = tf.minimum(cx + w/2.,img_width)
    # ymin = tf.maximum(cy - h/2.,0.)
    # ymax = tf.minimum(cy + w/2.,img_height)

#     conf = conf[:,:,0]
#     class_idx = class_idx[:,:,0]
#     coordinates = tf.transpose([class_idx, conf, ymin, xmin, ymax, xmax], [1,2,0])


#     #########################################################
#      #remove batch size. Here size is (nb_boxes/6)
#     coordinates = coordinates[0,:,:]
#     where = tf.cast(coordinates[:,0],tf.bool)
#     indices = tf.where(tf.reshape(where,[-1]))
#     coordinates = tf.transpose(tf.gather(coordinates, indices),[1,0,2])
    
#     coordinates = coordinates[0,:,:]
#     where = tf.cast(coordinates[:,1] - confidence_thresh,tf.bool)
#     indices = tf.where(tf.reshape(where,[-1]))
#     coordinates = tf.transpose(tf.gather(coordinates, indices),[1,0,2])

#     #########################################################

#     boxes = coordinates[0,:,2:6]
#     scores = coordinates[0,:,1]
#     class_idx = coordinates[0,:,0]
    
#     selected_boxes_idx = tf.image.non_max_suppression(boxes, scores, top_k, iou_threshold= iou_threshold)
#     selected_boxes = tf.gather(boxes, selected_boxes_idx)
#     selected_scores = tf.gather(scores, selected_boxes_idx)
#     selected_class_id = tf.gather(class_idx, selected_boxes_idx)

#     # selected_boxes = boxes
#     # selected_scores = scores
#     # selected_class_id = class_idx


#     xmin = selected_boxes[:,1]/float(img_width) #xmin
#     xmax = 1. - selected_boxes[:,3]/float(img_width)#xmax
#     ymin = selected_boxes[:,0]/float(img_height)#ymin
#     ymax = 1. - selected_boxes[:,2]/float(img_height) #ymax

#     selected_boxes = tf.transpose([ymin, xmin, ymax, xmax], [1,0])


#     final_pred = tf.concat((tf.expand_dims(selected_class_id,axis=1), 
#                            tf.expand_dims(selected_scores, axis=1), selected_boxes), axis=1)

#     final_pred = tf.expand_dims(final_pred, axis=0) #expand for batch size 

#     return final_pred


def decode_y2_tf(y_pred, 
                  confidence_thresh=0.5,
                  iou_threshold=0.45,
                  top_k='all',
                  input_coords='centroids',
                  normalize_coords=False,
                  img_height=None,
                  img_width=None):
    #y_pred_tensor = tf.stack(y_pred)
    y_pred_tensor = y_pred
    #print (sess.run(y_pred_tensor))
    y_pred_converted_tensor = y_pred_tensor[:,:,-14:-8]
    conf, idx = tf.nn.top_k(y_pred_tensor[:,:,:-12])
    class_idx = tf.cast(idx, tf.float32)
    pred_4_1 = tf.exp(tf.multiply(y_pred_converted_tensor[:,:,4], y_pred_tensor[:,:,-2]))
    pred_5_1 = tf.exp(tf.multiply(y_pred_converted_tensor[:,:,5], y_pred_tensor[:,:,-1]))
    pred_4_2 = tf.add(pred_4_1,(tf.multiply(pred_4_1, y_pred_tensor[:,:,-6])))
    pred_5_2 = tf.add(pred_5_1, (tf.multiply(pred_5_1, y_pred_tensor[:,:,-5])))
    pred_2_2 = tf.add(y_pred[:,:,-8], 
                        tf.multiply(y_pred_converted_tensor[:,:,2], 
                                           tf.multiply(y_pred_tensor[:,:,-4], y_pred_tensor[:,:,-6])))
    pred_3_2 = tf.add(y_pred[:,:,-7], 
                        tf.multiply(y_pred_converted_tensor[:,:,3], 
                                               tf.multiply(y_pred_tensor[:,:,-3], y_pred_tensor[:,:,-5])))
    cx =  pred_2_2
    cy =  pred_3_2
    w = pred_4_2
    h = pred_5_2
    xmin = tf.maximum(tf.subtract(cx, tf.div(w, 2.)), 0)
    xmax = tf.minimum(tf.add(cx, tf.div(w, 2.)), img_width-1)
    ymin = tf.maximum(tf.subtract(cy, tf.div(h, 2.)), 0)
    ymax = tf.minimum(tf.add(cy, tf.div(h, 2.)), img_height-1)
    conf = conf[:,:,0]
    class_idx = class_idx[:,:,0]
    coordinates = tf.transpose([class_idx, conf, ymin, xmin, ymax, xmax], [1,2,0])
    #coordinates = tf.transpose([class_idx, conf, ymin, xmin, ymax, xmax], [1,2,3,4,5,0])
    
    coordinates = coordinates[0,:,:]
    where = tf.cast(coordinates[:,0],tf.bool)
    indices = tf.where(tf.reshape(where,[-1]))
    coordinates = tf.transpose(tf.gather(coordinates, indices),[1,0,2])
    
    coordinates = coordinates[0,:,:]
    where = tf.cast(coordinates[:,1] - confidence_thresh,tf.bool)
    indices = tf.where(tf.reshape(where,[-1]))
    coordinates = tf.transpose(tf.gather(coordinates, indices),[1,0,2])
  
    '''    
    coordinates = coordinates[0,:,:]
    #print sess.run(tf.shape(coordinates))
    zero = tf.constant(0, dtype= tf.float32)
    where = tf.not_equal(coordinates[:,0], zero)
    indices = tf.where(where[:,])
    coordinates = tf.transpose(tf.gather(coordinates, indices),[1,0,2])
    coordinates = coordinates[0,:,:]
    thresh = tf.constant(confidence_thresh, dtype= tf.float32)
    where = tf.greater(coordinates[:,1], thresh)
   
    indices = tf.where(where[:,])
    
    coordinates = tf.transpose(tf.gather(coordinates, indices),[1,0,2])
    '''
    boxes = coordinates[0,:,2:6]
    scores = coordinates[0,:,1]
    class_idx = coordinates[0,:,0]
    #print 'boxes' , boxes.get_shape()
    
    selected_boxes_idx = tf.image.non_max_suppression(boxes, scores, top_k, iou_threshold= iou_threshold)
    selected_boxes = tf.gather(boxes, selected_boxes_idx)
    selected_scores = tf.gather(scores, selected_boxes_idx)
    selected_class_id = tf.gather(class_idx, selected_boxes_idx)
    
    #for normalization 
    # ymin = selected_boxes[:,0] / img_height
    # xmin = selected_boxes[:,0] / img_width
    # ymax = 1- selected_boxes[:,0] / img_height
    # xmax = 1- selected_boxes[:,0] / img_width
    # selected_boxes = tf.transpose([ymin, xmin, ymax, xmax], [1,0])

    xmin = selected_boxes[:,1]/float(img_width) #xmin
    xmax = 1. - selected_boxes[:,3]/float(img_width)#xmax
    ymin = selected_boxes[:,0]/float(img_height)#ymin
    ymax = 1. - selected_boxes[:,2]/float(img_height) #ymax
    selected_boxes = tf.transpose([ymin, xmin, ymax, xmax], [1,0])
    
    final_pred = tf.concat((tf.expand_dims(selected_class_id,axis=1), 
                           tf.expand_dims(selected_scores, axis=1), selected_boxes), axis=1)
    
 
    # use just 1 class animal
    
    # selected_class_id = selected_class_id -1
    # where = tf.cast(selected_class_id,tf.bool)
    # where = tf.logical_not(where)
    # indices = tf.where(tf.reshape(where, [-1]))
    # pred_animal = tf.gather(final_pred, indices)
    # pred_animal = pred_animal[:,0,:]
    # pred_animal = tf.expand_dims(pred_animal, axis=0)
    # selected_class_id = selected_class_id +1
    #animal
    indices = tf.where(tf.reshape(tf.logical_not(tf.cast(selected_class_id-1,tf.bool)), [-1]))
    pred_animal = tf.expand_dims(tf.gather(final_pred, indices)[:,0,:], axis =0)
    #building 
    indices = tf.where(tf.reshape(tf.logical_not(tf.cast(selected_class_id-2,tf.bool)), [-1]))
    pred_building = tf.expand_dims(tf.gather(final_pred, indices)[:,0,:], axis =0)
    #document
    indices = tf.where(tf.reshape(tf.logical_not(tf.cast(selected_class_id-3,tf.bool)), [-1]))
    pred_document = tf.expand_dims(tf.gather(final_pred, indices)[:,0,:], axis =0)
    #food 
    indices = tf.where(tf.reshape(tf.logical_not(tf.cast(selected_class_id-4,tf.bool)), [-1]))
    pred_food = tf.expand_dims(tf.gather(final_pred, indices)[:,0,:], axis =0)
    #moon
    indices = tf.where(tf.reshape(tf.logical_not(tf.cast(selected_class_id-5,tf.bool)), [-1]))
    pred_moon = tf.expand_dims(tf.gather(final_pred, indices)[:,0,:], axis =0)
    #person
    indices = tf.where(tf.reshape(tf.logical_not(tf.cast(selected_class_id-6,tf.bool)), [-1]))
    pred_person = tf.expand_dims(tf.gather(final_pred, indices)[:,0,:], axis =0)
    #sky 
    indices = tf.where(tf.reshape(tf.logical_not(tf.cast(selected_class_id-7,tf.bool)), [-1]))
    pred_sky = tf.expand_dims(tf.gather(final_pred, indices)[:,0,:], axis =0)
    #star
    indices = tf.where(tf.reshape(tf.logical_not(tf.cast(selected_class_id-8,tf.bool)), [-1]))
    pred_star = tf.expand_dims(tf.gather(final_pred, indices)[:,0,:], axis =0)
    #text 
    indices = tf.where(tf.reshape(tf.logical_not(tf.cast(selected_class_id-9,tf.bool)), [-1]))
    pred_text = tf.expand_dims(tf.gather(final_pred, indices)[:,0,:], axis =0)
    #final_pred = tf.expand_dims(final_pred, axis=0)
    final_pred = tf.concat([pred_animal, pred_building, pred_document, pred_food, pred_moon, pred_person, pred_sky, pred_star, pred_text],  axis = 1)
    return final_pred
    

class PostProcessing(Layer):

    def __init__(self,
                 img_height,                    
                 img_width,
                 confidence_thresh=0.5, 
                 iou_threshold=0.25, 
                 top_k=300, 
                 input_coords='centroids', 
                 normalize_coords=False,
                 n_boxes = 5,   
                 **kwargs):

       
        self.img_height = img_height
        self.img_width = img_width
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.input_coords = input_coords
        self.normalize_coords = normalize_coords
        self.n_boxes = n_boxes

        super(PostProcessing, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        We do not need any trainable weights in this layer 
        so we do not need to do self.add_weight , just calling the build function 
        '''
        self.input_spec = [InputSpec(shape = input_shape)]    
        super(PostProcessing, self).build(input_shape)

    def call(self, x, mask=None):
        
        
        '''
        Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, xmax, ymin, ymax]`.
        
        x: The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains

        '''

        predictions_decoded_tensor = decode_y2_tf(x,confidence_thresh=self.confidence_thresh, iou_threshold=self.iou_threshold, top_k=100,
                                                    input_coords='centroids', normalize_coords=False, 
                                                    img_height=self.img_height, img_width=self.img_width)
        

        #boxes_tensor = K.tile(K.constant(x1, dtype = 'float32'), (batch_size, n_boxes, predictions))
        #print "here..."    
        #print tf.shape(predictions_decoded_tensor)

        self.n_boxes = tf.shape(predictions_decoded_tensor)[1]
        k_tensor = tf.tile(predictions_decoded_tensor, (K.shape(x)[0], 1, 1))

        #k_tensor = tf.reduce_mean(k_tensor)
        return k_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_input_boxes, num_outputs = input_shape
        return (batch_size, self.n_boxes, num_outputs)    
        #return (batch_size, 1)    
