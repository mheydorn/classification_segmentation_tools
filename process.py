import os
import glob
import cv2
import tensorflow as tf
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt                     

#imagesDir = "/auto/shared/client_data/wada/sift_ai_images/SVW-1B8DFA/raw/"
#imagesDir = "imagesItScrewsUp/"
#imagesDir = "/data/tater_sai_no_sai/labeling/image"
#imagesDir = "/data/tater_pipeline/segmentation/oldSample"
#imagesDir = "/data/tater_sai_no_sai/noMaskRawImages"
#imagesDir = "/data/tater_sai_no_sai/yellowBelt"
#imagesDir = "/data/tater_pipeline/segmentation/raw_images_2018.Aug.14"
#imagesDir = "/auto/shared/client_data/wada/sorting_in_progress/lane10_2019_10/misshapen_raws"
#imagesDir = "lane_18_oct_2019"
#imagesDir = "/auto/shared/client_data/wada/mask_creating/validation/lane_18_oct_2019"
#imagesDir = "/auto/shared/client_data/wada/yellow_nov_2019/raw"
#imagesDir = "./test_images"
#imagesDir = "/auto/shared/client_data/wada/yellow_dec_03_2019"
#imagesDir = "/auto/shared/client_data/wada/"
#imagesDir = "/auto/shared/client_data/wada/bad_dec_18_2019/"
#imagesDir = "/auto/shared/client_data/wada/sift_ai_images/SVW-E255B6/runtime_raw_dec23/raw/"
#imagesDir = "/auto/shared/client_data/wada/potato_images/raws/yellow_potatoes_nov_2019/"
#imagesDir = "/auto/shared/client_data/wada/potato_images/raws/yellow_raws_dec_2019/"
imagesDir = "/auto/shared/client_data/wada/potato_images/raws/bad_dec_18_2019/"
from scipy.io import savemat, loadmat

#segmentModelFile = "models/segment_Sep_19.pb"
#segmentModelFile = "models/taters_segment_original.pb"
#segmentModelFile = "models/Sep24FirstGood.pb"
#segmentModelFile = "models/Sep26V1.pb" # first good
#segmentModelFile = "models/model1339.pb"
#segmentModelFile = "models/1459.pb"
#segmentModelFile = "models/1420.pb"
#segmentModelFile = "models/Nov21-731.pb"
#segmentModelFile = "models/atWadaDec30.pb"
segmentModelFile = "models/Dec23-1708-Segment.pb"


classifyModelFile = "models/Aug09V2.pb"
#classifyModelFile = "models/Dec03-816-Segment.pb"
#os.system("rm output/*")
def load_graph(frozen_graph_filename):
     with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())

     with tf.Graph().as_default() as graph:
         tf.import_graph_def(graph_def, name="prefix")
     return graph



def main():
    segmentGraph = load_graph(segmentModelFile)
    classifyGraph = load_graph(classifyModelFile)

   
    #files = 
    '''
    files = []
    with open("/home/svw/Downloads/blah") as fp:
        line = fp.readline()
        files.append(os.path.basename(line).split(".png")[0])
        while line:
           print(line)
           line = fp.readline()
           files.append(os.path.basename(line).split(".png")[0])

    files2 = []
    for f in files:
        if not f == "":
            files2.append("labeling/image/" + f + ".png")
    files = files2
    '''

    #files = files[0:250]
    for op in segmentGraph.get_operations():
        print(op.name)
        pass

    #for op in classifyGraph.get_operations():
    #    #print(op.name)
    #    pass


    #tensorlfow model
    #input_img_tensor_segment = segmentGraph.get_tensor_by_name('prefix/images_input:0')
    #output_img_tensor_segment = segmentGraph.get_tensor_by_name('prefix/Softmax:0')

    #input_img_tensor_segment = segmentGraph.get_tensor_by_name('prefix/input_1:0')
    #output_img_tensor_segment = segmentGraph.get_tensor_by_name('prefix/activation_1/Sigmoid:0')

    input_img_tensor_segment = segmentGraph.get_tensor_by_name('prefix/conv2d_1_input:0')
    output_img_tensor_segment = segmentGraph.get_tensor_by_name('prefix/Softmax/truediv:0')


    input_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/conv2d_13_input:0')
    output_img_tensor_classify = classifyGraph.get_tensor_by_name('prefix/activation_24/Softmax:0')



    goods = 0
    bads = 0
    clumps = 0

    with tf.Session(graph=segmentGraph) as sess:
        for imageNum, f in enumerate(glob.iglob(os.path.join(imagesDir, "*.png"))):
            imgSize = 224
            print (f)
            img_in = cv2.imread(f, 1)
            #img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

            img_in = img_in #/ 255.0
            img_original = np.copy(img_in)

            try:
                img_in = cv2.resize(img_in, (imgSize, imgSize))
            except:
                print("error resizing image")
                continue
            img_in = np.reshape(img_in, (1, imgSize, imgSize, 3))

            img_in = img_in.astype(np.float32)

            pred = sess.run(output_img_tensor_segment, feed_dict={ input_img_tensor_segment: img_in})
            #pred = pred.reshape((imgSize, imgSize, 1)) 
            #pred = sess.run(tf.nn.softmax(logits = pred, axis = 3))
            #sess.close()
        
            pred = pred[:,:,:,0]
            pred = np.squeeze(pred)

            #pred = pred[:,:, 0]
            pred = cv2.resize(pred, (img_original.shape[1],img_original.shape[0]))


            pred_thresh = np.copy(pred)
            pred_thresh[pred_thresh < 0.5] = 0
            pred_thresh[pred_thresh > 0.0001] = 1.0

            thresh = np.squeeze(pred_thresh).astype(np.uint8)

            #embed()
            thresh[thresh == 1] = 2
            thresh[thresh == 0] = 1
            thresh[thresh == 2] = 0

            #cv2.imshow("thresh", thresh.astype(np.uint8)*255) 
            #cv2.imshow("softmax", pred) 
            cv2.imwrite("masks/" + os.path.basename(f), thresh*255)
            #cv2.waitKey(0)



            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3,3))
           
            #embed()

            #Find contours and segment them all out

            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            

        
            if len(contours) > 1:
                #Remove nested contours
                contours2 = []

                for i, c in enumerate(contours):
                    if hierarchy [0,i,3] == -1:
                        contours2.append(c)

                contours = contours2


                
            masks = []
            originalSizeMasks = []
            allMasks = np.zeros(img_original.shape[0:2])
            for cnt in contours:
                mask = np.zeros(img_original.shape[0:2])
                mask = cv2.fillPoly(mask, [cnt], 1)  
                allMasks = cv2.fillPoly(allMasks, [cnt], 1)  

                originalSizeMasks.append(mask.copy())
                mask = mask.astype(np.float32)
                mask = cv2.resize(mask, (256, 256))
                #mask = np.zeros((256,256), dtype = np.uint8)
                #mask[128,128] = 1
                #cv2.imshow("before", mask.copy()*255)

                kernel = np.ones((3,3),np.uint8)
                #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 5, anchor = (-1, -1), borderType = cv2.BORDER_CONSTANT, borderValue = 0)

                #cv2.imshow("after",  mask.copy()*255)
                #cv2.waitKey(0)
        
                #mask = cv2.resize(mask.copy(), (1280, 616))
                #mask = cv2.resize(mask.copy(), (1264, 660))
                mask = cv2.resize(mask.copy(), (img_original.shape[1],img_original.shape[0]))
                masks.append(mask)  


            allContours = []
            potatoeCount = 0
            minX = 99999
            bestImage = {}
            for i, cnt in enumerate(contours):
                im = np.copy(img_original)
                idx=(masks[i]==0)
                im[idx]=[0,0,0]            

                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
        
                x,y,w,h = cv2.boundingRect(cnt)

                #img_original = cv2.rectangle(img_original,(x,y),(x+w,y+h),(0,255,0),2)


                # the order of the box points: bottom left, top left, top right, bottom right (x,y)
                if x <= 10 or x + w >= img_original.shape[1]-10:
                    continue

                
           

                boarderSize = 10
                box[0][0] -= boarderSize
                box[0][1] += boarderSize

                box[1][0] -= boarderSize
                box[1][1] -= boarderSize


                box[2][0] += boarderSize
                box[2][1] -= boarderSize


                box[3][0] += boarderSize
                box[3][1] += boarderSize

                width = int(rect[1][0] )
                height = int(rect[1][1] )
                if height * width < 3500:
                        continue

                aspectRatio = width / float(height)
                if aspectRatio < 1.0:
                    aspectRatio = aspectRatio / 1.0
                if aspectRatio > 4.0:
                    continue

                #cv2.imwrite("/data/tater_sai_no_sai/output/" + os.path.basename(f).split("png")[0] + "mat", originalSizeMasks[i]*255)

                src_pts = box.astype("float32")
                dst_pts = np.array([[0, height-1],
                                        [0, 0],
                                        [width-1, 0],
                                        [width-1, height-1]], dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # directly warp the rotated rectangle to get the straightened rectangle
                warped = cv2.warpPerspective(im, M, (width, height))


                if warped.shape[1] < warped.shape[0]:
                    warped = cv2.rotate(warped,  cv2.ROTATE_90_CLOCKWISE)
       
                warped_original = np.copy(warped)
                warped = cv2.resize(warped, (256, 256))
                #cv2.imshow("warped" + str(i), warped)
                #warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                warped = np.reshape(warped, (1, 256, 256, 3))

                warped = warped.astype(np.float32)
                warped = warped / 255.0

                

                #if x > minX:
                #    continue

                minX = x
                bestImage['img'] = warped_original
                bestImage['name'] = "output/" + os.path.basename(f)

                #cv2.imwrite("output/" + os.path.basename(f), warped_original)

                continue


                with tf.Session(graph=classifyGraph) as sess2:
                    pred_classify = sess2.run(output_img_tensor_classify, feed_dict={ input_img_tensor_classify: warped})
                    sess2.close()

                class_idx = np.argmax(pred_classify[0])

                if class_idx == 0:
                    bads += 1
                    color = [0,0,255]
                if class_idx == 1:
                    clumps += 1
                    color = [85, 186,211]
                if class_idx == 2:
                    goods += 1
                    color = [0,255,0]
                print(class_idx)
                print(pred_classify)
                #cv2.imwrite("output/" + os.path.basename(f), warped_original)
                bordersize = 15
                border = cv2.copyMakeBorder(warped_original, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=color )

                cv2.imwrite("output/" + os.path.basename(f), border)
                #embed()
                #exit()

                #cv2.imshow(str(potatoeCount), border)
                potatoeCount += 1
                #2 is good
                #0 is bad
                #1 is clump
         
            
            try:
                cv2.imwrite(bestImage['name'], bestImage['img'])
            except:
                print("none found")
            

            saveMat = False
            if saveMat:
                mat_contents = {}
                mat_contents["Hierarchy"] = np.array(["{{'Layer', {}}, {'Original', {}}}"], dtype='<U33')
                mat_contents["Layer"] =  allMasks.astype(np.uint32) #originalSizeMasks[i]*255

                mat_filename = "/data/tater_sai_no_sai/output/" + os.path.basename(f).split("png")[0] + "mat"
                savemat(mat_filename, mat_contents, do_compression=True)


            cv2.drawContours(img_original, contours, -1, (0,255,0), 3) 

            #print("Good ratio:", goods / float(goods + bads + clumps), "Bad ratio:", bads / float(goods + bads + clumps))
            
            #cv2.imshow("img_original",img_original)
            #cv2.imwrite("output/" + os.path.basename(f), img_original)
            #cv2.imshow("pred", pred[0][:,:,0])
            #cv2.imshow("thresh", thresh*255)
            
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #if imageNum > 200:
            #    break
        print ("Done")

main()
