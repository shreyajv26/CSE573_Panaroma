import argparse
import sys
import numpy as np
import cv2
import glob
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt
import random
import os

'''
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''

#---------------------extracting local invariant descriptors (SIFT) -------------------------

def detect_compute_keypoints(img):
    
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    
    #Detecting keypoints and descriptors
    kp, desc = sift.detectAndCompute(gray_img,None)      #kp will be a list of keypoints and desc is a numpy array of shape Number_of_Keypoints×128

    return kp,desc

#--------------------------KEYPOINT MATCHING - EUCLIDEAN DISTANCE -------------------------------------------------------

'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
'''
def calc_euclidean_dist(kp1,kp2,desc1,desc2):
    Y = cdist(desc1, desc2, 'sqeuclidean')      #Computes the squared Euclidean distance
    threshold = 9000                             #Setting the threshold as per pixel values
    
    image1_points = np.where(Y < threshold)[0]       #Finding points in image 1
    
    image2_points = np.where(Y < threshold)[1]       #Finding points in image 2
    
    #Finding coordinates
    coordinates_in_image1 = np.array([kp1[point].pt for point in image1_points])
    coordinates_in_image2 = np.array([kp2[point].pt for point in image2_points])
    
    good_points = (np.concatenate( (coordinates_in_image1, coordinates_in_image2) , axis=1 ))
    
    return good_points

#----------------------------RANSAC ALGORITHM ---------------------------------------------


'''
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#
https://salzis.wordpress.com/2014/06/10/robust-linear-model-estimation-using-ransac-python-implementation/

1: Select randomly the minimum number of points required to determine the model
parameters.
2: Solve for the parameters of the model.
3: Determine how many points from the set of all points fit with a predefined tolerance .
4: If the fraction of the number of inliers over the total number points in the set
exceeds a predefined threshold τ , re-estimate the model parameters using all the
identified inliers and terminate.
5: Otherwise, repeat steps 1 through 4 (maximum of N times)
'''


def ransac_algo(goodPoints,totalIteration):
    
    i = 0
    persp_transform_final = []
    highest_inlier_count = 0
    
    while i < totalIteration:
        i = i + 1
        # Select 4 points randomly
                
        Pair1 = random.SystemRandom().choice(goodPoints)
        Pair2 = random.SystemRandom().choice(goodPoints)
        Pair3 = random.SystemRandom().choice(goodPoints)
        Pair4 = random.SystemRandom().choice(goodPoints)
        
        Pairs=np.concatenate(([Pair1],[Pair2],[Pair3],[Pair4]),axis=0)
        
        # Finding homography matrix for this 4 matching pairs
        # H = get_homography(fourMatchingPairs)
        
        image1_points = np.float32(Pairs[:,0:2])
        image2_points = np.float32(Pairs[:,2:4])
        
        persp_transform = cv2.getPerspectiveTransform(image1_points, image2_points)
                
        points1 = np.concatenate( (goodPoints[:, 0:2], np.ones((len(goodPoints), 1))), axis=1)
        points2 = goodPoints[:, 2:4]
        
        #https://stackoverflow.com/questions/11053099/how-can-you-tell-if-a-homography-matrix-is-acceptable-or-not
        
        '''
        To find bad homographies apply found H to your original points and see the separation from your expected points that is |x2-H*x1| < Tdist, where Tdist is your threshold for distance error. If there are only few points that satisfy this threshold your homography may be bad and you probably violated one of the above mentioned requirements.
Reference has also been taken from GITHUB codes to find the error in the homography matrix.

'''
        #https://culturalengineerassociation.weebly.com/uploads/8/6/7/7/86776910/programming_computer_vision_with_python.pdf
        #http://laid.delanover.com/homography-estimation-explanation-and-python-implementation/
        # Calculate error in the matrix above -- Sampson error
        
        Points = np.zeros((len(goodPoints), 2))
        for k in range (len(goodPoints)):
            mult = np.matmul(persp_transform, points1[k])
            #ValueError: could not broadcast input array from shape (3) into shape (2)
            Points[k] = (mult/mult[2])[0:2]
            
        error = np.linalg.norm(points2 - Points, axis=1) ** 2
        #error = cdist(points2, Points, 'sqeuclidean')
        #print(error)
        indices = np.where(error < 0.9)[0]
        inliers = goodPoints[indices]
        
        if (len(inliers)) > highest_inlier_count:
            highest_inlier_count = len(inliers)
            
            persp_transform_final = persp_transform.copy()
        
    return persp_transform_final

#-----------------------------------------------------------------------------------------------------------------

def main():
    
    program = sys.argv[0]
    directory = sys.argv[1]
    imageDir  = directory + '/*.jpg'
    
	#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
	#https://github.com/pavanpn/Image-Stitching/blob/master/stitch_images.py
    
    images = [cv2.imread(file,1) for file in glob.glob(imageDir)]

    #Stitch 2 images
    
    if len(images)==2:

#--------------------2 images Stitch -----------------------------------------------------------------------------
        # SIFT feature detection
        kp1,desc1 = detect_compute_keypoints(images[0])
        kp2,desc2 = detect_compute_keypoints(images[1])

        # Stitching image 1 and image 2
        pertran12 = ransac_algo(calc_euclidean_dist(kp1,kp2, desc1,desc2), 2000)
        
        warp = cv2.warpPerspective(images[0], pertran12 ,(int(images[0].shape[1] + images[1].shape[1]*0.5),int(images[0].shape[0] + images[1].shape[0]*0.5)))
            
        warp[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]
        cv2.imwrite( directory + '/panorama.jpg', warp)
    
#----------------------------------------------------------------------------------------------------------------------------
    else:
        # Stitch 3 or more images
        
        kp1, desc1 = detect_compute_keypoints(images[0])
        kp2, desc2 = detect_compute_keypoints(images[1])
        kp3, desc3 = detect_compute_keypoints(images[2])

        euc_dist12 = calc_euclidean_dist(kp1,kp2,desc1,desc2)
        euc_dist13 = calc_euclidean_dist(kp1,kp3,desc1,desc3)
        euc_dist23 = calc_euclidean_dist(kp2,kp3,desc2,desc3)

        '''
        My code was behaving inconsistently for different images. For some images stitching was working properly, for others it was not.
      So I checked on Google and saw a post which suggested stitching on both left and right sides and then choose the best one.
      I have implemented the same by checking the number of black pixels.
        
        '''
#----------------------Left Stitch--------------------------------------------------------
        
        #https://kushalvyas.github.io/stitching.html
        
        pertran12 = ransac_algo(euc_dist12, 1000)

        warp = cv2.warpPerspective(images[0], pertran12 ,( int(images[0].shape[1] + images[1].shape[1]*0.5),int(images[0].shape[0] + images[1].shape[0]*0.5)))

        warp[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]

        cv2.imwrite( directory + '/panorama12.jpg', warp)

        # Stitching this image with image 3 now
        image12 = cv2.imread(directory + '/panorama12.jpg',1)
        
        kp12, desc12 = detect_compute_keypoints(image12)
        euc_dist12 = calc_euclidean_dist(kp12,kp3,desc12,desc3)
                
        pertran23 = ransac_algo(euc_dist12, 1000)

        warp = cv2.warpPerspective(image12, pertran23,(int(image12.shape[1] + images[2].shape[1]*0.5),int(image12.shape[0] + images[2].shape[0]*0.5)))

        warp[0:images[2].shape[0], 0:images[2].shape[1]] = images[2]

        cv2.imwrite( directory + '/panorama1.jpg', warp)
        os.remove(directory + '/panorama12.jpg')
        
        #count the number of black pixels
        black = np.zeros(3)
        panaroma1 = cv2.imread(directory + '/panorama1.jpg', 1)
        count1 = 0
        width = panaroma1.shape[0]
        height = panaroma1.shape[1]
        
        for i in range(width):
            for j in range(height):
                val = panaroma1[i, j, :]
                if np.array_equal(val, black):
                    count1 = count1 + 1

#---------------------Right Stitch---------------------------------------------------------

        # Stitching image 3 and image 2
        euc_dist32 = calc_euclidean_dist(kp3,kp2,desc3,desc2)
        pertran32 = ransac_algo(euc_dist32, 1000)

        warp = cv2.warpPerspective(images[2],pertran32,(int(images[2].shape[1] + images[1].shape[1]*0.5),int(images[2].shape[0] + images[1].shape[0]*0.5)))

        warp[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]

        cv2.imwrite( directory + '/panorama32.jpg', warp)

        # Stitching this image and image 1 now
        image32 = cv2.imread(directory + '/panorama32.jpg',1)
        
        kp32, desc32 = detect_compute_keypoints(image32)
        
        euc_dist321 = calc_euclidean_dist(kp32,kp1,desc32,desc1)
        pertran321 = ransac_algo(euc_dist321, 1000)

        warp = cv2.warpPerspective(image32, pertran321,(int(image32.shape[1] + images[0].shape[1]*0.5),int(image32.shape[0] +images[0].shape[0]*0.5)))
            
        warp[0:images[0].shape[0], 0:images[0].shape[1]] = images[0]

        cv2.imwrite( directory + '/panorama2.jpg', warp)
        os.remove(directory + '/panorama32.jpg')
        
        #count the number of black pixels
        
        panaroma2 = cv2.imread(directory + '/panorama2.jpg', 1)
        count2 = 0
        width = panaroma2.shape[0]
        height = panaroma2.shape[1]
        for i in range(width):
            for j in range(height):
                val = panaroma2[i, j, :]
                if np.array_equal(val, black):
                    count2 = count2 + 1
        
# -------------------------------------------------------------------------------------------------------------------------  
        
        if count1 < count2:
            os.remove(directory + '/panorama2.jpg')
            os.rename(directory + '/panorama1.jpg', directory +'/panorama.jpg')
        else:
            os.remove(directory + '/panorama1.jpg')
            os.rename(directory + '/panorama2.jpg',directory + '/panorama.jpg')
        
if __name__ == "__main__":
    main()










