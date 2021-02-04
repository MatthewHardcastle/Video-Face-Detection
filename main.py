import cv2
#opencv documentation
#https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html?highlight=detectmultiscale

video = cv2.VideoCapture("face_Clip.mp4")
Face_Classifier = "Face.xml" #Face classifier xml file-this xml will check our image and check it against the data and if it passed then it will be classed as a Face
FaceCheck = cv2.CascadeClassifier(Face_Classifier)

# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

while True:#As we are reading frames from a video it is better to use a while loop so it can go through each frame until the video ends
    (read_status, frame) = video.read()#This will capture the current frame from the video and call it frame and the read status returns if the read was successful
    if read_status:
        Grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break#if the read was unsuccessful break out of the while loop this will happen at the end of the video

    FaceTracker = FaceCheck.detectMultiScale(Grayscale)#this will apply the Face classifier to the image and then will give out  coordinates for each Face found
#if i were to print this then it would give the coordinates of the Face
    for(X, Y,W, H) in FaceTracker:#This will take the coordinates given in Face tracker and then from that it will draw a rectangle to so the Face position
        cv2.rectangle(frame, (X, Y), (X+W, Y+H), (255, 0, 0), 2)  #This is drawing the rectangle from the given coordinates from each frame

    cv2.imshow('Face detection',frame)#opens the file that we just made and calls the window "Face detection"
    cv2.waitKey(1)#stops the window from autoclosing-closes when its the end of the video
