import cv2

videoCaptureObject = cv2.VideoCapture(0)
ctr = 0
while(True):
    ret,frame = videoCaptureObject.read()
    cv2.imshow('Capturing Video',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        print(ctr)
        cv2.imwrite("./customdata2/B/"+str(ctr)+".jpg", frame)
        ctr += 1
        print(ctr)
    if(ctr == 600):
        videoCaptureObject.release()
        cv2.destroyAllWindows()
        break