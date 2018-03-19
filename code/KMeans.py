import numpy as np 
import tensorflow as tf  
from random import shuffle

def KMeans(inputImg, clusterNum):
    """
    inputImg is a numpy array, inputImg's size is m * n, m is the number of images, n is the size of each image's features
    clusterNum is an integer value smaller than m
    """

    np.array(inputImg)
    imgNum, imgSize = inputImg.shape
    
    assert clusterNum <= imgNum

    # initialize a new graph and set it as default to avoid occupation of previous call
    graph = tf.Graph()
    with graph.as_default():

        sess = tf.Session()

        # initialize centroids from raw input
        shuffleIndex = [i for i in range(imgNum)]
        shuffle(shuffleIndex)
        centroids = [tf.Variable(inputImg[shuffleIndex[i]]) for i in range(clusterNum)]

        distances = tf.Variable(tf.zeros([clusterNum, imgNum], tf.float64))
        lastClass = tf.Variable(tf.zeros([imgNum], tf.int32))
        imgClass = tf.Variable(tf.zeros([imgNum], tf.int32))

        init = tf.global_variables_initializer()
        sess.run(init)

        ###test
        ###print ("initial centroids: ", sess.run(centroids))

        # store present img class state in lastClass
        storeClass = tf.assign(lastClass, imgClass)

        # define two placeholder to calculate distance
        # dist1 is used to hold all image features
        # dist2 is used to hold one centroid value 
        dist1 = tf.placeholder(tf.float64, shape=(imgNum, imgSize))
        dist2 = tf.placeholder(tf.float64, shape=(1, imgSize))
        # calculate euclide distance
        calculateDist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(dist1, dist2), 2), 1))

        # update distances
        updateDistances = []
        for i in range (clusterNum):
            updateDistances.append(tf.assign(distances[i], calculateDist))
        ##distanceValue = tf.placeholder(tf.float64)
        ##assignDist = tf.assign(distances, distanceValue)

        # find the minist distance for each image and decide class belonging
        getImgClass = tf.assign(imgClass, tf.cast(tf.argmin(distances, axis=0), tf.int32))

        # calculate new centroids
        points = tf.placeholder(tf.float64, shape=[None, imgSize])
        newCentroid = tf.reduce_mean(points, 0)
        # update centroids
        updateCentroids = []
        for i in range(clusterNum):    
            updateCentroids.append(tf.assign(centroids[i], newCentroid))
        
        # loop and run graph until converge
        cnt = 0
        while cnt < 1000:
            # store old class 
            sess.run(storeClass)
            
            # calculate all distances for each cluster
            # update distances
            for i in range(clusterNum):
                sess.run(updateDistances[i], feed_dict={dist1: inputImg, dist2: np.reshape(sess.run(centroids[i]), (1,imgSize))})
            ###print ("this is the ",cnt, "th loop, and centroids are:", sess.run(centroids))
            ###print ("the distances matrix is: ", sess.run(distances))
            
            # get class of each image
            sess.run(getImgClass)
            
            # check whether new class is equal to last class
            currentClassValue = sess.run(imgClass)
            lastClassValue = sess.run(lastClass)
            ###if (currentClassValue == lastClassValue).all():
            ###    break
            ###print ("the current class vector is: ", currentClassValue)
            if cnt == 0:
                print ("the first cluster division is : ", currentClassValue)
            cnt += 1
            
            # calculate new centroids one by one
            for i in range(clusterNum):
                tempCluster = np.zeros((1,imgSize), dtype=float)
                for j in range (imgNum):
                    if currentClassValue[j] == i:
                        tempCluster = np.append(tempCluster, np.atleast_2d(inputImg[j]), 0)
                sess.run(updateCentroids[i], feed_dict={points: tempCluster[1:,:]})
                ###print ("the ",i, "th time update centroids, and now centroids matrix is: ", sess.run(centroids))
        
        print ("total loop: ", cnt)
        return sess.run([centroids, imgClass])

            