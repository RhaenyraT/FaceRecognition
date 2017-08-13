
import cv2
import cv2.cv as cv
import numpy as np
import fnmatch
import sys
import os

# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
directory='C:/Users/Poori/Desktop/KULEUVEN/KULApoorva/CV/3'

for filename in fnmatch.filter(os.listdir(directory),'*.jpg'):
    file_in= directory+"/"+filename
    file_out= directory+"/train/"+filename
    img = cv2.imread(file_in)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    cv2.imshow('img',gray)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        newimg=gray[y:y+h,x:w+x]
        cv2.imshow("Faces found", img)
        cv2.waitKey(0)
        cv2.imshow("Crop",newimg)
        cv2.waitKey(0)
        resized_image = cv2.resize(newimg, (100, 100)) 
        cv2.imshow("Cropresize",resized_image)
        cv2.waitKey(0)
        cv2.imwrite(file_out,resized_image)

nbDim=10000
filenames = fnmatch.filter(os.listdir(directory+"/train/"),'*.jpg')
nbImages = len(filenames)
X = np.zeros( (nbImages,nbDim) )#, dtype=np.uint8 )
for i,filename in enumerate( filenames ):
    file_in1 = directory+"/train/"+filename
    img = np.asarray(cv2.imread(file_in1))
    #print file_in1
    #print img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    X[i,:] = gray.flatten()
    #print X.shape

def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    
    X -= mu 
    return np.dot(W.T, X) #TODO

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    
    return np.dot(W, Y) + mu #TODO

def pca(X, nb_components=0):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    
    [n,d] = X.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n

    #TODO 
    
    mu = X.mean(axis=0)
    
    for i in range(n):
        X[i,:] -= mu

    Hermitian_matrix = (np.dot(X, X.T) / float(n))
    eigenvalues, eigenvectors = np.linalg.eigh(Hermitian_matrix)

	# changing the sign of negative eigen value and changing the direction of corresponding eigen vector (as only absolute value/variance is of interest)  
    negEig = np.where(eigenvalues < 0)
    eigenvalues[negEig] = eigenvalues[negEig] * -1.0
    eigenvectors[:,negEig] = eigenvectors[:,negEig] * -1.0

	# sorting the components in order of decreasing variance
    sorted_indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvectors = eigenvectors[:,sorted_indexes][:,0:nb_components]

    eigenvectors = np.dot(X.T, eigenvectors)

	# normalizing eigenvectors 
    for i in range(nb_components):
    	eigenvectors[:,i] = eigenvectors[:,i] / np.linalg.norm(eigenvectors[:,i])

    return (eigenvalues, eigenvectors, mu)
    
def normalize(img):
    '''
    Normalize an image such that it min=0 , max=255 and type is np.uint8
    '''
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)

if __name__ == '__main__':
    #create database of normalized images
    for directory in ["data/arnold", "data/barack"]: 
        create_database(directory, show = False)
    
    show = True
    
    #create big X arrays for arnold and barack
    Xa = createX("data/arnold2")
    Xb = createX("data/barack2")
            
    #Take one part of the images for the training set, the rest for testing
    nbTrain = 6
    Xtest = np.vstack( (Xa[nbTrain:,:],Xb[nbTrain:,:]) )
    Ctest = ["arnold"]*(Xa.shape[0]-nbTrain) + ["barack"]*(Xb.shape[0]-nbTrain)
    Xa = Xa[0:nbTrain,:]
    Xb = Xb[0:nbTrain,:]

    #do pca
    [eigenvaluesa, eigenvectorsa, mua] = pca(Xa,nb_components=6)    
    [eigenvaluesb, eigenvectorsb, mub] = pca(Xb,nb_components=6)
    
    #visualize
    cv2.imshow('img',np.hstack( (mua.reshape(100,100),
                                 normalize(eigenvectorsa[:,0].reshape(100,100)),
                                 normalize(eigenvectorsa[:,1].reshape(100,100)),
                                 normalize(eigenvectorsa[:,2].reshape(100,100)))
                               ).astype(np.uint8))
    cv2.waitKey(0) 
    cv2.imshow('img',np.hstack( (mub.reshape(100,100),
                                 normalize(eigenvectorsb[:,0].reshape(100,100)),
                                 normalize(eigenvectorsb[:,1].reshape(100,100)),
                                 normalize(eigenvectorsb[:,2].reshape(100,100)))
                               ).astype(np.uint8))
    cv2.waitKey(0) 
            
    nbCorrect = 0
    for i in range(Xtest.shape[0]):
        X = Xtest[i,:]
        
        #project image i on the subspace of arnold and barack
        Ya = project(eigenvectorsa, X, mua )
        Xa= reconstruct(eigenvectorsa, Ya, mua)
        
        Yb = project(eigenvectorsb, X, mub )
        Xb= reconstruct(eigenvectorsb, Yb, mub)
        if show:
            #show reconstructed images
            cv2.imshow('img',np.hstack( (X.reshape(100,100),
                                         np.clip(Xa.reshape(100,100), 0, 255),
                                         np.clip(Xb.reshape(100,100), 0, 255)) ).astype(np.uint8) )
            cv2.waitKey(0)   

        #classify i
        if np.linalg.norm(Xa-Xtest[i,:]) < np.linalg.norm(Xb-Xtest[i,:]):
            bestC = "arnold"
        else:
            bestC = "barack"
        print str(i)+":"+str(bestC)
        if bestC == Ctest[i]:
            nbCorrect+=1
    
    #Print final result
    print str(nbCorrect)+"/"+str(len(Ctest))





