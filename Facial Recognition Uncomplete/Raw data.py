import os 
import time 
import uuid 
import cv2
from numpy import number

##?????From her block out cause ive already got stuff labeled 
##this section gives me data to train detection
image_path = os.path.join('data', 'images')
number_images = 45

cap = cv2.VideoCapture(0)
for imgenum in range(number_images):
    print('Collecting image {}'.format(imgenum))
    ret, frame = cap.read()
    imgename = os.path.join(image_path, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgename, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        break
cap.release()
cv2.destroyAllWindows()



### blocking stops here 