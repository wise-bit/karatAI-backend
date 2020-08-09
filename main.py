import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


# Specify the paths for the 2 files
protoFile = "./pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./pose/mpi/pose_iter_160000.caffemodel"

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip',
                    'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']


# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

threshold = 0.2
total_points = 18


# Read image
# frame = cv2.imread("karate1.jpg") # best example
frame = cv2.imread("./test/photo.jpg")
# Specify the input image dimensions
width, height, channels = frame.shape
# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height), (0, 0, 0), swapRB=False, crop=False)
# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

net.setInput(inpBlob)
output = net.forward()

keypoints = []

for point_index in range(total_points):

    probMap = output[0, point_index, :, :]
    probMap = cv2.resize(probMap, (height, width))

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
    mapMask = np.uint8(mapSmooth>threshold)
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

print(keypoints)

# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.imshow(mapMask, alpha=0.6)
# plt.show()

# show plot

fig, ax = plt.subplots()
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
for point in keypoints:
    ax.add_patch(plt.Circle(point[:2], 4, color='b', alpha=0.8))
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()


def image_resize():
    return None

