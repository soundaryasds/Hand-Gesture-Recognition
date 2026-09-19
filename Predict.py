# for building CNN architechture
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
# Numpy to interpret the image as binary array
import numpy as np
# For performing operations on image
from PIL import Image
import cv2
# For resizing the image
import imutils

# Setting an empty background to be updated later
background = None

# To reduce the computation complexity we are reducing the image size
def resize(image):
    width = 100
    img = Image.open(image)
    wpercent = (width/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((width,hsize), Image.ANTIALIAS)
    img.save(image)

# We are recognizing the background by averaging the frames
def average(image, avg_weight):
    
    global background
    # Create the background once it is stable
    if background is None:
        background = image.copy().astype("float")
        return

    # Weighted average of the background
    cv2.accumulateWeighted(image, background, avg_weight)

# Background elimination process to separate the foreground from the background
def segmentation(image, threshold=25):
    global background
    # Difference between the background and foreground (hand region)
    diff = cv2.absdiff(background.astype("uint8"), image)

    # Binary thresholding the difference to obtain the foreground
    # To convert the gray pixels in the hand region to white and to get a uniform binary image.
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # Finding the contours in the thresholded image (To recognize the hand region)
    # Contours are a curve joining all the continuous points (forming a boundary), having same color or intensity. 
    # Contours are a used for analyzing the shape and detecting and recognizing the object.
    (_,contours,_) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # When no contours are found exit
    if len(contours) == 0:
        return
    else:
        # Using the contour area, calculated the hand_region (maximum contour)
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    # Weight for finding the average for background
    avg_weight = 0.5

    # To open the Camera
    cam = cv2.VideoCapture(0)

    # Coordinates to target the hand region in the frame
    top=10
    right=350
    bottom=225
    left=590
    # Initialize the no of frames
    frames = 0
    # Specify whether to start or stop capturing the frames
    start = False

    #Until quitting 
    while(True):
        # Current frame
        (captured, frame) = cam.read()

        # Resize frame
        frame = imutils.resize(frame, width = 700)

        # Flip to avoid inverted view from camera
        frame = cv2.flip(frame, 1)

        # Copy the current frame
        copy = frame.copy()
        
        # Split the frame's dimensions to height and width
        (height, width) = frame.shape[:2]

        # Looking for the hand region
        region = frame[top:bottom, right:left]

        # Converting the frame to grayscale
        grayscale = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Gaussian Blurring
        # To remove the random noise present in the image we have used the median blur with a kernel size of 7 to smooth the image in the frame
        grayscale = cv2.GaussianBlur(grayscale, (7, 7), 0)

        # Calibrating the frames to detect a change in gesture 
        if frames < 30:
            average(grayscale, avg_weight)
        else:
            # Separating the hand from the background
            hand_region = segmentation(grayscale)
            
            if hand_region is not None:
                # Split the values from hand region into a tuple
                (thresholded, segmented) = hand_region

                # Display the segmented hand region separately as a frame
                cv2.drawContours(copy, [segmented + (right, top)], -1, (0, 0, 255))
                if start:
                    # Create a file with the captured frame
                    cv2.imwrite('Capture.png', thresholded)
                    resize('Capture.png')
                    # Predicting the class and probability of the class for the curr
                    prediction_result, probability = predictGesture()
                    showResults(prediction_result, probability)
                cv2.imshow("Theshold", thresholded)

        # To display the live feed create a window
        cv2.rectangle(copy, (left, top), (right, bottom), (0,255,0), 2)
        
        frames += 1

        # Display the current frame
        cv2.imshow('Video Feed', copy)

        # User input
        key = cv2.waitKey(1) & 0xFF

        # To Quit capturing
        if key == ord('q'):
            break
            
        # To start capturing
        if key == ord('s'):
            start = True

def predictGesture():
    # To Predict the class of each gesture using the CNN model
    image = cv2.imread('Capture.png')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Check with the layers of CNN ang get the probable class of gesture
    prediction_result = model.predict([gray_img.reshape(89, 100, 1)])
    # Calculating the probability of the class by using the binary array of the image
    return np.argmax(prediction_result), (np.amax(prediction_result) / (prediction_result[0][0] + prediction_result[0][1] + prediction_result[0][2]))

def showResults(prediction_result, probability):

    text = np.zeros((300,512,3), np.uint8)
    classification = ""
    
    if prediction_result == 0:
        classification = "Swing"
    elif prediction_result == 1:
        classification = "Palm"
    elif prediction_result == 2:
        classification = "Fist"
    
    # Displaying the predicted class as text on the image
    cv2.putText(text,"Predicted Class : " + classification, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Displaying the probability of the class on the image
    cv2.putText(text,"Probability : " + str(probability) , (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Statistics :", text)

# Creating CNN Model
tf.reset_default_graph() # Clears the default graph stack and resets the global default graph.

# The input_data is a layer that will be used as the input layer to our network. 
network=input_data(shape=[None,89,100,1],name='input')

# Designing a Conv2D wrapper consisting of bias along with relu activation
# relu removes all the negative values from the convolution and all the positive values remain the same but all the negative values get changed to zero
# Max Pooling returns the maximum value from the portion of the image covered by the Kernel and performs as a Noise Suppressant. 
# It discards the noisy activations altogether and also performs de-noising along with dimensionality reduction.

# Building 2D convolutional layer into architecture of CNN.

# Creating Convolution Layer

network=conv_2d(network,32,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)
network=conv_2d(network,64,2,activation='relu')
network=max_pool_2d(network,2)

network=conv_2d(network,128,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

network=conv_2d(network,256,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

network=conv_2d(network,256,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

network=conv_2d(network,128,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

network=conv_2d(network,64,2,activation='relu')
# Max Pooling (down-sampling)
network=max_pool_2d(network,2)

# Creating Fully connected layer
network=fully_connected(network,1000,activation='relu')

# Dropout is a regularization method that approximates training a large number of neural networks with different architectures in parallel.
# Apply Dropout
network=dropout(network,0.75)

# Fully connected layer
network=fully_connected(network,3,activation='softmax')

# With TFLearn estimators
# Define loss and optimizer
network=regression(network,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

# Train the Network using Classifier
model=tflearn.DNN(network,tensorboard_verbose=0)

# Loading the trained model
model.load("TrainedModel/GestureRecogModel.tfl")

main()
