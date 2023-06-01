import sys
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

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


#define histogram constant parameters
H_BINS = 40
S_BINS = 55
HISTOGRAM_SIZE = [H_BINS, S_BINS]
H_RANGES = [0, 180]
S_RANGES = [0, 256]
RANGES = H_RANGES + S_RANGES
CHANNELS = [0, 1]

#define new object probability as constant
NEW_OBJECT_PROBABILITY = 0.4

def createBipartiteGraphMatrix(numberOfBboxPreviousFrame, numberOfBboxCurrentFrame, previousBboxImages, currentBboxImages):

    # init the matrix so its columns are objects previous frame, and rows are object from current frame
    # we need to extend the matrix columns by number of objects on current frame 
    # - becous every object on current frame may appear for the first time
    matrix = np.ones((numberOfBboxCurrentFrame, numberOfBboxPreviousFrame + numberOfBboxCurrentFrame))

    #init matrix so every element will have probability of the new object
    matrix = matrix * NEW_OBJECT_PROBABILITY

    # object similarity is calculated based on image hue and saturation histogram correlation
    for indexPrevious, bBoxImgPrevious in enumerate(previousBboxImages):

        histprevious = cv2.calcHist([bBoxImgPrevious], CHANNELS, None, HISTOGRAM_SIZE, RANGES, accumulate=False)
        cv2.normalize(histprevious, histprevious, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        for indexCurrent, bBoxImgCurrent in enumerate(currentBboxImages):
            histcurrent = cv2.calcHist([bBoxImgCurrent], CHANNELS, None, HISTOGRAM_SIZE, RANGES, accumulate=False)
            cv2.normalize(histcurrent, histcurrent, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            score = cv2.compareHist(histprevious, histcurrent, 0)

            # fill the matrix with similarity score of corresponding objects
            matrix[indexCurrent, indexPrevious] = score

    return matrix

if __name__ == '__main__':

    #get data
    dataDirectory = sys.argv[1]
    boundingBoxesfileName = dataDirectory + '/bboxes.txt'

    framesData = getFramesDataToList(boundingBoxesfileName)
    firstFrame = framesData[0]
    firstFrameBboxNum = firstFrame[1]

    #for first image every Bbox will be "new" -> so its index is -1
    outputStringFirstFrame = ""
    for bboxInd in range(firstFrameBboxNum):
        outputStringFirstFrame += "-1"
    
    outputStringFirstFrame = outputStringFirstFrame.rstrip()
    print(outputStringFirstFrame)

    #start processing for all the images -> main loop of the code
    for frameIndex in range(len(framesData) - 1):
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
        croppedPreviousImgs = []
        for bBoxImg in previousBboxImgs:
            croppedPreviousImgs.append(cropImageByFraction(bBoxImg, 1/3.2, 1/3.5))
        
        previousBboxImgs = croppedPreviousImgs

        croppedCurrentImgs = []
        for bboxImg in currentBboxImgs:
            croppedCurrentImgs.append(cropImageByFraction(bboxImg, 1/3.2, 1/3.5))

        currentBboxImgs = croppedCurrentImgs

        # declare probability matrix (that will be representing the bipartite graph) 
        # where rows are current frame objects and columns are previous frame objects -> the elements are the probability that
        # object x from previous frame is the same object as object y on current frame 
        bipartiteGraphMatchingMatrix = createBipartiteGraphMatrix(previousBboxNum, currentBboxNum, previousBboxImgs, currentBboxImgs)
        
        # get optimal solution to find best boxes matches in bipartite graph using Hungarian Method
        rowIndexes, colIndexes = linear_sum_assignment(bipartiteGraphMatchingMatrix, maximize=True)

        # get indexes of the bounding boxes
        bBoxIndexes = []
        for rowIndex, colIndex in zip(rowIndexes, colIndexes):
            #if index of responding previous boundingBox pointing to "new" objects - its pointing to the extended row ow 
            # matrix - greater or equal number of number of bounding boxes on previous frame
            # -> treat as a new object so index is -1
            if (colIndex >= previousBboxNum):
                bBoxIndexes.append(-1)
            else:
                # if element of the matrix differs from probability of new object -> the index of current bBox will be the index 
                # of corresponding object from previous frame 
                bBoxIndexes.append(colIndex)

        #converting indexes of bBoxes to correct format  
        outputString = ""
        for index in bBoxIndexes:
            outputString += str(index) + " "

        #print using rstrip to eliminate spaces at the end of string
        outputString = outputString.rstrip()
        print(outputString)

