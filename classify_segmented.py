import os
import glob
import cv2
import tensorflow as tf
from IPython import embed
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

put_bad_threshold_in_name = True

#imagesDir = "ValidationData/mostly_good"
#imagesDir = "ValidationData/yellow_bad_Jan_14/"
#imagesDir = "/data/tater_sai_no_sai/ValidationData/validation_lane_15_feb_19/run/good"
#imagesDir = "/data/tater_sai_no_sai/ValidationData/validation_lane_15_feb_19/run/single"
#imagesDir = "/data/tater_sai_no_sai/ValidationData/validation_lane_15_feb_19/run/single"
#imagesDir = "/data/tater_pipeline/classification/Latest_Dataset/test_images/mud"
#imagesDir = "/auto/shared/client_data/wada/potato_images/raws/mud_jan_6_2020/raw"
#imagesDir = "/data/tater_sai_no_sai/segment_raw_output/segmented"
imagesDir = "/data/tater_pipeline/classification/Latest_Dataset/skin_check"
#imagesDir = "/data/tater_sai_no_sai/ValidationData/single"

#classifyModelFile = "models/August19V1.pb"
#classifyModelFile = "models/August19V1.pb"
#classifyModelFile = "models/835.pb"
#classifyModelFile = "models/1622.pb"
#classifyModelFile = "models/Dec09-1516.pb"
#classifyModelFile = "models/1622.pb"
#classifyModelFile = "models/Dec17-1600.pb"
#classifyModelFile = "/data/tater_pipeline/classification/Feb13-1606.pb"
classifyModelFile = "/data/tater_pipeline/keras_to_tensorflow/Feb18-941.pb"

def load_graph(frozen_graph_filename):
     with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())

     with tf.Graph().as_default() as graph:
         tf.import_graph_def(graph_def, name="prefix")
     return graph

bads = 0
goods = 0
clumps = 0

os.system("rm -rf classify_segmented_output")
os.system("mkdir classify_segmented_output")
os.system("mkdir  classify_segmented_output/bad")
os.system("mkdir  classify_segmented_output/good")
os.system("mkdir  classify_segmented_output/clump")
os.system("mkdir  classify_segmented_output/skin_check")

classifyGraph = load_graph(classifyModelFile)

for op in classifyGraph.get_operations():
    print(op.name)


f= open("hist.txt","w+")

#Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'old_bruise': 5, 'rub': 6}
#835 Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'no_netting_good': 5, 'old_bruise': 6, 'rub': 7}
#922 Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'new_bruise': 5, 'no_netting_good': 6, 'old_bruise': 7, 'rub': 8}
#Dec09-1516.pb Class indexes are {'clump': 0, 'frozen': 1, 'good': 2, 'green': 3, 'growth_crack': 4, 'misshapen': 5, 'new_bruise': 6, 'no_netting_good': 7, 'old_bruise': 8, 'rub': 9, 'yellow_bad': 10, 'yellow_good': 11}
#Feb13-1606.pb class_indexes = {'clump': 0, 'frozen': 1, 'good': 2, 'green': 3, 'growth_crack': 4, 'misshapen': 5, 'mud': 6, 'new_bruise': 7, 'no_netting_good': 8, 'old_bruise': 9, 'rub': 10, 'skin_check': 11, 'yellow_bad': 12, 'yellow_good': 13}

def decideClass(pred_classify, thresh):

    if len(pred_classify) == 7:
        good_probability = pred_classify[1] +  pred_classify[6]
    elif len(pred_classify) == 8:
        good_probability = pred_classify[1] + pred_classify[5] + pred_classify[7]
    elif len(pred_classify) == 9:
        good_probability = pred_classify[1] + pred_classify[6] + pred_classify[8]
    elif len(pred_classify) == 12:
        good_probability = pred_classify[2] + pred_classify[7] + pred_classify[9] + pred_classify[11]
    elif len(pred_classify) == 14:
        good_probability = pred_classify[2] + pred_classify[8] + pred_classify[10] + pred_classify[13] + pred_classify[6]
    else:
        print("Classify model not supported - stopping\n")
        exit()

    other_index = 11

    clump_probability = pred_classify[0]
    bad_probability = 1.0 - (clump_probability + good_probability)

    classes = [clump_probability, good_probability, bad_probability]
    my_class = np.argmax(classes)
    
    '''
    if pred_classify[11] > 0.2:
        return "bad", pred_classify[11],pred_classify[11], pred_classify[11]
    else:
        return "good", pred_classify[11],pred_classify[11], pred_classify[11]
    '''

    '''
    if pred_classify[11] > 0.5:
        return "skin_check", bad_probability, good_probability, clump_probability, pred_classify[other_index]
    '''

    if bad_probability > thresh:   
        return "bad", bad_probability, good_probability, clump_probability, pred_classify[other_index]
    else:
        if clump_probability > good_probability:
            return "clump", bad_probability, good_probability, clump_probability, pred_classify[other_index]
        else:
            return "good", bad_probability, good_probability, clump_probability, pred_classify[other_index]

with tf.Session(graph=classifyGraph) as sess:
    for i, f in enumerate(glob.iglob(os.path.join(imagesDir, "*.png"))):

        #input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_13_input:0')
        #output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_24/Softmax:0')

        #input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_9_input:0')
        #output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_18/Softmax:0')

        #input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_1_input:0')
        #output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_6/Softmax:0')

        #835
        #input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_1_input:0')
        #output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_6/Softmax:0')

        input_name = ""
        output_name = ""
        for op in classifyGraph.get_operations():
            name = op.name
            if "input" in name:
                input_name = name
            if "Softmax" in name:
                output_name = name
        if input_name == "" or output_name == "":
            print("Could not find input or output tensor names. Exiting.")
            exit()

        input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_1_input:0')
        output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_6/Softmax:0')

        try:
            img = cv2.imread(f, 1)
            img_original = np.copy(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            aspectRatio = img.shape[0]/img.shape[1]
            if aspectRatio < 1.0:
                aspectRatio = 1.0 / aspectRatio

            if aspectRatio > 2:
                continue
            img = cv2.resize(img, (256, 256))

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.reshape(img, (1, 256, 256, 3))

            img = img.astype(np.float32)
            img = img / 255.0
        except:
            continue
        
        pred_classify = sess.run(output_img_tensor_classify, feed_dict={ input_img_tensor_classify: img})
        pred_classify = pred_classify[0]
        print(pred_classify)

        base = os.path.basename(f)
        name = base
        classDecision, bad_probability, good_probability, clump_probability, other = decideClass(pred_classify, thresh = 0.5)

        if put_bad_threshold_in_name:
            name = "other="+"{:0.4f}".format(other) + "__good=" + "{:0.4f}".format(good_probability) + "__clump=" + "{:0.4f}".format(clump_probability) + "__bad=" + "{:0.4f}".format(bad_probability) + "__" + name

        if classDecision != "clump" and classDecision != "skin_check":
            classDecision = "good"

        if classDecision == "bad":#Bad
            bads += 1
            color = [0,0,255]
            cv2.imwrite("classify_segmented_output/bad/" + name, img_original)
 
        if classDecision == "clump":#clump
            clumps += 1
            color = [85, 186,211]
            cv2.imwrite("classify_segmented_output/clump/" + name, img_original)
        if classDecision == "good":#Good
            goods += 1
            color = [0,255,0]
            cv2.imwrite("classify_segmented_output/good/" +  name, img_original)

        if classDecision == "skin_check":#Good
            goods += 1
            color = [0,255,0]
            cv2.imwrite("classify_segmented_output/skin_check/" +  name, img_original)

        













