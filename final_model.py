import cv2
import time
import numpy as np
from random import randint
import os
import glob
import json

required_points = [7, 6, 1, 3, 4, 11, 8, 12, 9, 13, 10, 0]
final_keypoints = []
json_list = []
meta_json = {}


def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)


def clean_directory():
    files = glob.glob('./frames/*')
    for f in files:
        os.remove(f)


clean_directory()
vidcap = cv2.VideoCapture('./test/cut_karate_1.mp4')  # Insert video
success, image = vidcap.read()
count = 0

while success:

    meta_json["timestamp"] = 500 * count

    # cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
    try:
        success, image1 = vidcap.read()
        success, image1 = vidcap.read()
        success, image1 = vidcap.read()
        success, image1 = vidcap.read()
        success, image1 = vidcap.read()
        count += 1
        print(count)
    except:
        print("failing...")
        data = json.dumps(json_list)
        with open('emergency_data.json', 'w') as outfile:
            json.dump(data, outfile)
        continue

    # image1 = cv2.imread("./test/photo.jpg")#Add according to your own path
    protoFile = "./pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 18

    # COCO Output Format
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip',
                        'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

    POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                  [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                  [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

    # index of pafs correspoding to the POSE_PAIRS
    # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
    mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
              [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
              [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]

    colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
              [0, 255, 0], [255, 200, 100], [255, 0, 255], [
                  0, 255, 0], [255, 200, 100], [255, 0, 255],
              [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

    def getKeypoints(probMap, threshold=0.1):
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth > threshold)
        keypoints = []
        # find the blobs
        contours, _ = cv2.findContours(
            mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
        return keypoints

    # Find valid connections between the different joints of a all persons present
    def getValidPairs(output):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        # loop for every POSE_PAIR
        for k in range(len(mapIdx)):
            # A->B constitute a limb
            try:
                pafA = output[0, mapIdx[k][0], :, :]
                pafB = output[0, mapIdx[k][1], :, :]
                pafA = cv2.resize(pafA, (frameWidth, frameHeight))
                pafB = cv2.resize(pafB, (frameWidth, frameHeight))
            except:
                continue
            # Find the keypoints for the first and second limb
            candA = detected_keypoints[POSE_PAIRS[k][0]]
            candB = detected_keypoints[POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)
            # If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid
            if(nA != 0 and nB != 0):
                valid_pair = np.zeros((0, 3))
                for i in range(nA):
                    max_j = -1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                               pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores)/len(paf_scores)
                        # Check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                        if(len(np.where(paf_scores > paf_score_th)[0])/n_interp_samples) > conf_th:
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(
                            valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)
                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else:  # If no keypoints are detected
                #print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])
        # print(valid_pairs)
        return valid_pairs, invalid_pairs

    # This function creates a list of keypoints belonging to each person
    # For each detected valid pair, it assigns the joint(s) to a person
    def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
        # the last number in each row is the overall score
        personwiseKeypoints = -1 * np.ones((0, 19))
        for k in range(len(mapIdx)):
            if k not in invalid_pairs:
                try:
                    # partAs = valid_pairs[k][:,0]
                    partAs = [elems[0] for elems in valid_pairs[k]]
                    # partBs = valid_pairs[k][:,1]
                    partBs = [elems[1] for elems in valid_pairs[k]]
                except:
                    continue
                indexA, indexB = np.array(POSE_PAIRS[k])
                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break
                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(
                            int), 2]+valid_pairs[k][i][2]
                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(keypoints_list[valid_pairs[k]
                                                     [i, :2].astype(int), 2])+valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack(
                            [personwiseKeypoints, row])
        return personwiseKeypoints
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]
    print("width: " + str(frameWidth) + ", " + "height: " + str(frameHeight))
    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)
    inpBlob = cv2.dnn.blobFromImage(
        image1, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    print("Time Taken in forward pass = {}".format(time.time() - t))
    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = 0.1
    # temp_count = 0
    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        # print("Keypoints - {} : {} --- {}".format(keypointsMapping[part], keypoints, temp_count))
        # temp_count+=1

        # if keypointsMapping[part]=='Nose':
        #     print(len(keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1
        detected_keypoints.append(keypoints_with_id)

    # frameClone = image1.copy()
    frameClone = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    meta_json["coordinates"] = {}

    ii = 0

    for index in required_points:
        meta_json["coordinates"][ii+1] = {}

        if (len(detected_keypoints[index])):
            final_keypoints.append(detected_keypoints[index][0][0:2])
            meta_json["coordinates"][ii +
                                     1]["x"] = detected_keypoints[index][0][0]
            meta_json["coordinates"][ii +
                                     1]["y"] = detected_keypoints[index][0][1]
        else:
            final_keypoints.append(None)
            meta_json["coordinates"][ii+1]["x"] = 0
            meta_json["coordinates"][ii+1]["y"] = 0
        ii += 1

    meta_json["coordinates"][13] = {}
    # print(final_keypoints)
    try:
        hip_l_x = meta_json["coordinates"][12]["x"]
        hip_l_y = meta_json["coordinates"][12]["y"]
        hip_r_x = meta_json["coordinates"][9]["x"]
        hip_r_y = meta_json["coordinates"][9]["y"]
        meta_json["coordinates"][13]["x"], meta_json["coordinates"][13]["y"] = midpoint(
            hip_l_x, hip_l_y, hip_r_x, hip_r_y)
    except:
        meta_json["coordinates"][13]["x"], meta_json["coordinates"][13]["y"] = None, None

    print(json.dumps(meta_json))
    json_list.append(meta_json)

    # for i in range(nPoints):
    #     for j in range(len(detected_keypoints[i])):
    #         # cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, [255, 0, 0], -1, cv2.LINE_AA)
    #         cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

    # valid_pairs, invalid_pairs = getValidPairs(output)
    # personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

    # cv2.imshow("Testing GUI Frame 1 - Test 2", frameClone)
    # # cv2.imshow("Testing GUI Frame 1 - Test 2", cv2.cvtColor(frameClone, cv2.COLOR_BGR2GRAY))

    # cv2.waitKey(0)

print("count: " + str(count))
# print(json.dumps(json_list))
data = json.dumps(json_list)
with open('metadata.json', 'w') as outfile:
    json.dump(data, outfile)
