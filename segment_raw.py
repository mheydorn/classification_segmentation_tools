import os
import glob
import cv2
import tensorflow as tf
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt                     
from scipy.io import savemat, loadmat

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
#imagesDir = "/auto/shared/client_data/wada/potato_images/raws/bad_dec_18_2019/"
imagesDir = "ValidationData/raw/"

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

write_mask = True
write_raw_with_contours = True
write_segmented = True
write_mat_mask = True
os.system("rm -rf segment_raw_output")
os.system("mkdir segment_raw_output")
os.system("mkdir segment_raw_output/masks")
os.system("mkdir segment_raw_output/raw_with_contours")
os.system("mkdir segment_raw_output/segmented")
os.system("mkdir segment_raw_output/mat_masks")

def load_graph(frozen_graph_filename):
     with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())

     with tf.Graph().as_default() as graph:
         tf.import_graph_def(graph_def, name="prefix")
     return graph

def main():
    segmentGraph = load_graph(segmentModelFile)

    for op in segmentGraph.get_operations():
        print(op.name)
        pass

    input_img_tensor_segment = segmentGraph.get_tensor_by_name('prefix/conv2d_1_input:0')
    output_img_tensor_segment = segmentGraph.get_tensor_by_name('prefix/Softmax/truediv:0')


    with tf.Session(graph=segmentGraph) as sess:
        for imageNum, f in enumerate(glob.iglob(os.path.join(imagesDir, "*.png"))):
            imgSize = 224
            print (f)
            img_in = cv2.imread(f, 1)

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
        
            pred = pred[:,:,:,0]
            pred = np.squeeze(pred)

            pred = cv2.resize(pred, (img_original.shape[1],img_original.shape[0]))

            pred_thresh = np.copy(pred)
            pred_thresh[pred_thresh < 0.5] = 0
            pred_thresh[pred_thresh >= 0.5] = 1.0

            thresh = np.squeeze(pred_thresh).astype(np.uint8)
            thresh[thresh == 1] = 2
            thresh[thresh == 0] = 1
            thresh[thresh == 2] = 0

            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3,3))

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

                kernel = np.ones((3,3),np.uint8)
        
                mask = cv2.resize(mask.copy(), (img_original.shape[1],img_original.shape[0]))
                masks.append(mask)  

            allContours = []
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

                src_pts = box.astype("float32")
                dst_pts = np.array([[0, height-1],
                                        [0, 0],
                                        [width-1, 0],
                                        [width-1, height-1]], dtype="float32")

                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(im, M, (width, height))

                if warped.shape[1] < warped.shape[0]:
                    warped = cv2.rotate(warped,  cv2.ROTATE_90_CLOCKWISE)
       
                warped_original = np.copy(warped)
                warped = cv2.resize(warped, (256, 256))
                warped = np.reshape(warped, (1, 256, 256, 3))

                warped = warped.astype(np.float32)
                warped = warped / 255.0

                minX = x
                bestImage['img'] = warped_original
                bestImage['name'] = os.path.basename(f)
         
            if write_segmented:
                try:
                    cv2.imwrite(os.path.join("segment_raw_output/segmented", bestImage['name']), bestImage['img'])
                except:
                    print("none found")
            
            if write_mat_mask:
                mat_contents = {}
                mat_contents["Hierarchy"] = np.array(["{{'Layer', {}}, {'Original', {}}}"], dtype='<U33')
                mat_contents["Layer"] =  allMasks.astype(np.uint32)

                mat_filename = os.path.join("segment_raw_output/mat_masks", os.path.basename(f).split("png")[0] + "mat")
                savemat(mat_filename, mat_contents, do_compression=True)

            if write_mask:
                cv2.imwrite(os.path.join("segment_raw_output/masks", os.path.basename(f)), (allMasks*255).astype(np.uint8))


            cv2.drawContours(img_original, contours, -1, (0,255,0), 3) 

            if write_raw_with_contours:
                cv2.imwrite(os.path.join("segment_raw_output/raw_with_contours", os.path.basename(f)), img_original)
                
        print ("Done")

main()
