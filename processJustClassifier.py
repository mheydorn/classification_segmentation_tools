import os
import glob
import cv2
import tensorflow as tf
from IPython import embed
import numpy as np

#imagesDir = "/auto/shared/temp/run/all/*"
#imagesDir = "/auto/shared/temp/run/all/*"
#imagesDir = "output/*"
#imagesDir = "ValidationData/*"
#imagesDir = "Lane15Validation/SVW-CFEB92/*"
#imagesDir = "lane14/clumps/*"
#imagesDir = "strangeOne/*.png"
#imagesDir = "yellow_potatoes_Dec03_good_segmentation/good_segmentation/*"
#imagesDir = "yellow_dec13_not_good_lane1/*"
#imagesDir = "/data/tater_pipeline/classification/Dataset_Sep_04/green_copy/*"
#imagesDir = "ValidationData/mostly_good/*"
#imagesDir = "/auto/shared/client_data/wada/potato_images/raws/bad_dec_18_2019/*"
imagesDir = "output/*7835*"

#classifyModelFile = "models/August19V1.pb"
#classifyModelFile = "models/August19V1.pb"
#classifyModelFile = "models/835.pb"

#classifyModelFile = "models/1622.pb"
#classifyModelFile = "models/Dec09-1516.pb"

classifyModelFile = "models/1622.pb"
#classifyModelFile = "models/Dec17-1600.pb"

#Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'old_bruise': 5, 'rub': 6}

#835 Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'no_netting_good': 5, 'old_bruise': 6, 'rub': 7}
#922 - 1622 Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'new_bruise': 5, 'no_netting_good': 6, 'old_bruise': 7, 'rub': 8}

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

os.system("rm  classification_test_output/bad/*")
os.system("rm  classification_test_output/good/*")
os.system("rm  classification_test_output/clump/*")

classifyGraph = load_graph(classifyModelFile)

for op in classifyGraph.get_operations():
    print(op.name)


f= open("hist.txt","w+")

#922 Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'new_bruise': 5, 'no_netting_good': 6, 'old_bruise': 7, 'rub': 8}
#835 Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'no_netting_good': 5, 'old_bruise': 6, 'rub': 7}
#Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'old_bruise': 5, 'rub': 6}
#Dec09-1516.pb Class indexes are {'clump': 0, 'frozen': 1, 'good': 2, 'green': 3, 'growth_crack': 4, 'misshapen': 5, 
#'new_bruise': 6, 'no_netting_good': 7, 'old_bruise': 8, 'rub': 9, 'yellow_bad': 10, 'yellow_good': 11}

#Class indexes are {'clump': 0, 'frozen': 1, 'good': 2, 'green': 3, 'growth_crack': 4, 'misshapen': 5, 'new_bruise': 6, 'no_netting_good': 7, 'old_bruise': 8, 'rub': 9, 'yellow_bad': 10, 'yellow_good': 11}

'''
    float clump_probability = vals[0];
    float good_probability = vals[2] + vals[7] + vals[9] + vals[11];
    float bad_probability = vals[1] + vals[3] + vals[4] + vals[5] +  vals[6] + vals[8] + vals[10];

'''

def decideClass(pred_classify, thresh):
    clump_probability = pred_classify[0]
    #good_probability = pred_classify[2] +  pred_classify[7]  + pred_classify[9] + pred_classify[11]
    good_probability = pred_classify[1] +  pred_classify[6]  + pred_classify[8] 
    bad_probability = 1.0 - (clump_probability + good_probability)

    embed()
    classes = [clump_probability, good_probability, bad_probability]
    my_class = np.argmax(classes)

    #bad_prob_normalized = 
    #f.write(bad
    
    if bad_probability > thresh:   
        return "bad", bad_probability
    else:
        if clump_probability > good_probability:
            return "clump", bad_probability
        else:
            return "good", bad_probability
    

with tf.Session(graph=classifyGraph) as sess:
    for i, f in enumerate(glob.glob(imagesDir)):

        #input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_13_input:0')
        #output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_24/Softmax:0')

        #input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_9_input:0')
        #output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_18/Softmax:0')

        #input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_1_input:0')
        #output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_6/Softmax:0')

        #835
        #input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_1_input:0')
        #output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_6/Softmax:0')

        #922
        input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_1_input:0')
        output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_6/Softmax:0')


        try:
            img = cv2.imread(f, 1)
            img_original = np.copy(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.reshape(img, (1, 256, 256, 3))

            img = img.astype(np.float32)
            img = img / 255.0
        except:
            continue
        
        
        
        pred_classify = sess.run(output_img_tensor_classify, feed_dict={ input_img_tensor_classify: img})


        pred_classify = pred_classify[0]

        
        print(len(pred_classify))
        #exit()

        #Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'old_bruise': 5, 'rub': 6}

        #835 Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'no_netting_good': 5, 'old_bruise': 6, 'rub': 7}
        #922 Class indexes are {'clump': 0, 'good': 1, 'green': 2, 'growth_crack': 3, 'misshapen': 4, 'new_bruise': 5, 'no_netting_good': 6, 'old_bruise': 7, 'rub': 8}

        #Dec09-1516.pb Class indexes are {'clump': 0, 'frozen': 1, 'good': 2, 'green': 3, 'growth_crack': 4, 'misshapen': 5, 
        #'new_bruise': 6, 'no_netting_good': 7, 'old_bruise': 8, 'rub': 9, 'yellow_bad': 10, 'yellow_good': 11}

        '''
        threshs = ""
        threshs += "clump=" + "{:.2f}".format(pred_classify[0])+ " "
        threshs += "frozen=" + "{:.2f}".format(pred_classify[1])+ " "
        threshs += "good=" + "{:.2f}".format(pred_classify[2])+ " "
        threshs += "green=" + "{:.2f}".format(pred_classify[3])+ " "
        threshs += "growth_crack=" + "{:.2f}".format(pred_classify[4])+ " "
        threshs += "misshapen=" + "{:.2f}".format(pred_classify[5])+ " "
        threshs += "new_bruise=" + "{:.2f}".format(pred_classify[6])+ " "
        threshs += "no_netting_good=" + "{:.2f}".format(pred_classify[7])+ " "
        threshs += "old_bruise=" + "{:.2f}".format(pred_classify[8])+ " "
        threshs += "rub=" + "{:.2f}".format(pred_classify[9])+ " "
        threshs += "yellow_bad=" + "{:.2f}".format(pred_classify[10])+ " "
        threshs += "yellow_good=" + "{:.2f}".format(pred_classify[11])+ " "
        '''


        base = os.path.basename(f)

        #name = threshs + "-" + base +".png"
        name = base


        
        classDecision, bad_probability = decideClass(pred_classify, thresh = 0.50)

        classDecision = "good"

        name = str(bad_probability) + "__" + name
        if classDecision == "bad":#Bad
            bads += 1
            color = [0,0,255]
            cv2.imwrite("classification_test_output/bad/" + name, img_original)

               
        if classDecision == "clump":#clump
            clumps += 1
            color = [85, 186,211]
            cv2.imwrite("classification_test_output/clump/" + name, img_original)
        if classDecision == "good":#Good
            goods += 1
            color = [0,255,0]
            cv2.imwrite("classification_test_output/good/" +  name, img_original)

        print(pred_classify)
        #embed()













