import sys
import os
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from munkres import Munkres, print_matrix


# read data from to file to list where the structure is [imgName, numberOfBoxes, [boundingBoxesData]] 
# -> the last element of the list are the lists of Bboxes features (x,y,w,h) ex. ['c6s1_087551.jpg', 2, [[197, 55, 32, 130], [55, 111, 38, 135]]]
def getFramesDataToList(boundingBoxesInfoFileName):
    #initialize list of data from all frames
    allFramesData = []
    #open the file with informations about bBoxes in read mode
    with open(boundingBoxesInfoFileName, 'r') as fh:
        while True:
            #init single frame data
            frameData = []
            #read frame file name
            frameName = fh.readline().strip('\n') 
            #break if frameName not found / eof
            if not frameName:  
                break

            frameData.append(frameName)

            #get number of bounding boxes on previous image
            numberOfBoxes = int(fh.readline().strip('\n'))
            frameData.append(numberOfBoxes)

            #get data about bBoxes localisation on previous image
            boxesData = []
            for boxData in range(numberOfBoxes):
                boxLine = fh.readline().strip('\n')
                boxDataStringList = boxLine.split(" ")
                #convert data to int - so it could be compatible with opencv img data
                boxDataIntList = [ int(float(x)) for x in boxDataStringList ]
                boxesData.append(boxDataIntList)
            
            frameData.append(boxesData)

            allFramesData.append(frameData)

        # sort data by image name
        allFramesData = sorted(allFramesData, key=lambda x: x[0])

        return allFramesData


def getBoundingBoxImagesHSV(image, bBoxesCoordinates):
    bBoxImages = []
    for box in bBoxesCoordinates:
        x,y,w,h = box
        bBoxImg = image[y: y+h, x:x+w]
        bBoxImg = cv2.cvtColor(bBoxImg, cv2.COLOR_BGR2HSV)
        bBoxImages.append(bBoxImg)
    
    return bBoxImages


def cropImageByFraction(image, fractionX, fractionY):
    imgHeight, imgWidth, _ = image.shape
    cutPixelsHorizontal = int(fractionX * imgWidth/2)
    cutPixelsVertical = int(fractionY * imgHeight/2)
    image = image[0+cutPixelsVertical:imgHeight-cutPixelsVertical, 0+cutPixelsHorizontal: imgWidth-cutPixelsHorizontal]

    return image


if __name__ == '__main__':

    #get data
    dataDirectory = sys.argv[1]
    boundingBoxesfileName = dataDirectory + '/bboxes.txt'

    framesData = getFramesDataToList(boundingBoxesfileName)
    firstFrame = framesData[0]
    firstFrameBboxNum = firstFrame[1]

    #for first image every Bbox will be "new"
    outputStringFirstFrame = ""
    for bboxInd in range(firstFrameBboxNum):
        outputStringFirstFrame += "-1"
    
    print(outputStringFirstFrame)


    #start processing for all the images -> main loop of the code
    for frameIndex in range(len(framesData) - 1):
        cv2.destroyAllWindows()
        # load previous frame data
        previousFrame = framesData[frameIndex]
        previousImg = cv2.imread('data/frames/' + previousFrame[0])
        previousBboxNum = previousFrame[1]
        previousBboxes = previousFrame[2]

        # load current frame data
        currentFrame = framesData[frameIndex+1]
        currentImg = cv2.imread('data/frames/' + currentFrame[0])
        currentBboxNum = currentFrame[1]
        currentBboxes = currentFrame[2]

        #get images containing only bounding boxes from both frames
        #using the HSV space to compare hue and saturation channels later
        previousBboxImgs = getBoundingBoxImagesHSV(previousImg, previousBboxes)
        currentBboxImgs = getBoundingBoxImagesHSV(currentImg, currentBboxes)

        #crop the BboxImages so that it does not take to the background into calculation
        for bBoxImg in previousBboxImgs:
            bBoxImg = cropImageByFraction(bBoxImg, 1/7, 1/7)
        
        for bboxImg in currentBboxImgs:
            bboxImg = cropImageByFraction(bboxImg, 1/7, 1/7)


        # displayBoxes
        for index, img in enumerate(previousBboxImgs):
            cv2.imshow('previous' + str(index), cv2.cvtColor(img, cv2.COLOR_HSV2BGR))

        for index, img in enumerate(currentBboxImgs):
            cv2.imshow('current' + str(index), cv2.cvtColor(img, cv2.COLOR_HSV2BGR))


        #define histogram constant parameters
        H_BINS = 50
        S_BINS = 60
        HISTOGRAM_SIZE = [H_BINS, S_BINS]
        H_RANGES = [0, 180]
        S_RANGES = [0, 256]
        RANGES = H_RANGES + S_RANGES
        CHANNLES = [0, 1]

        #define new object probability as constant
        NEW_OBJECT_PROBABILITY = 0.48

        #declare probability matrix where rows are current frame objects and columns are previous frame objects
        scoresMatrix = np.ones((currentBboxNum, previousBboxNum + currentBboxNum))
        scoresMatrix = scoresMatrix * NEW_OBJECT_PROBABILITY

        print("@@@@@@@@@@@@@@@@@@")
        print(currentFrame[0])

        for indexprevious, bBoxImgprevious in enumerate(previousBboxImgs):

            histprevious = cv2.calcHist([bBoxImgprevious], CHANNLES, None, HISTOGRAM_SIZE, RANGES, accumulate=False)
            cv2.normalize(histprevious, histprevious, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            for indexcurrent, bBoxImgcurrent in enumerate(currentBboxImgs):

                histcurrent = cv2.calcHist([bBoxImgcurrent], CHANNLES, None, HISTOGRAM_SIZE, RANGES, accumulate=False)
                cv2.normalize(histcurrent, histcurrent, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                score_x = cv2.compareHist(histprevious, histcurrent, 0)

                score = score_x

                scoresMatrix[indexcurrent, indexprevious] = score
                
                # print("previous" + str(indexprevious) + " to current"+str(indexcurrent) +": ", score)

        print("-----------------")
        print("MATRIX: \n", scoresMatrix)

        row_ind, col_ind = linear_sum_assignment(scoresMatrix, maximize=True)

        bBoxIndexes = []
        for index in range(len(row_ind)):
            if (scoresMatrix[row_ind[index], col_ind[index]] == 0.48):
                bBoxIndexes.append(-1)
            else:
                bBoxIndexes.append(col_ind[index])
                
        
        outputString = ""
        for index in bBoxIndexes:
            outputString += str(index) + " "

        print(outputString.rstrip())


        
        


        

        key = ord(' ')
        while (key != ord('x')):
            key = cv2.waitKey(10)
        



    