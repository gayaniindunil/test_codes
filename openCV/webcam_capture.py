import cv2
import numpy as np 
import sys
# cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)

print ('Showing camera feed. Click window or press any key to stop.')
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    white_boder = frame.copy()
    white_boder[0:102,:,:] = (255,0,0) #top
    white_boder[-100:-1,:,:] = (255,0,0) #bottom
    white_boder[:,0:100,:] = (255,0,0) #right
    white_boder[:,-100:-1,:] = (255,0,0) #left

    cv2.imshow('MyWindow', white_boder)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    success, frame = cameraCapture.read()
    cv2.imwrite('id_001.jpg',white_boder)

cv2.destroyWindow('MyWindow')
cameraCapture.release()


# fps = videoCapture.get(cv2.CAP_PROP_FPS)
# size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

# success, frame = videoCapture.read()

# while success: # Loop until there are no more frames.
#     videoWriter.write(frame)
#     success, frame = videoCapture.read()