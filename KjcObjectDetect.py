# By  KJC

# with the speak function


import tensorflow.keras
import numpy as np
import cv2
# import pyttsx3
import math
import time


# # set the speaker parameters
# speaker = pyttsx3.init()
# voices = speaker.getProperty('voices')
# speaker.setProperty('voice', voices[1].id)
# speaker.setProperty('rate', 200)


# # set a speak function
# def speak(arg):
#     speaker.say(arg)
#     speaker.runAndWait()

# disable scientific notation for clarity
np.set_printoptions(suppress=False)

# load the model
model = cv2.CascadeClassifier('KJCmodel.h5')
# model = tensorflow.keras.models.load_model('KJCmodel.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


# set the camera
cam = cv2.VideoCapture(0)

while cam.isOpened():

    # setting the image framework
    check, frame = cam.read()

    cv2.imwrite('scan.jpg', frame)

    img = cv2.imread('scan.jpg')
    img = cv2.resize(img, (224, 224))

    imageArray = np.asarray(img)

    # normalize the image
    normalizedImage = (imageArray.astype(np.float32)/127.0) - 1
    data[0] = normalizedImage

    # prediction
    prediction = model.predict(data)
    # print(prediction)

    coordinates = model.detectMultiScale2(frame)


    # for m, wm, bg, bk, bt, ct, ch, cp, ctn, dg, dr, fd, gl, ht, ot, pn, ph, pl, sf, tb, t, tc, tr, tv, vl, ws,wh, wn in prediction:

    #     pr = max(m, wm, bg, bk, bt, ct, ch, cp, ct, dg, dr, fd, gl, ht, ot, pn, ph, pl, sf, tb, t, tc, tr, tv, vl, ws,wh, wn)
    #     print(pr)

    #     if pr == m:
    #         text = 'man'
    #         pass
    #     if pr == wm:
    #         text = 'woman'
    #         pass
    #     if pr == bg:
    #         text = 'bag'
    #         pass
    #     if pr == bk:
    #         text = 'book'
    #         pass
    #     if pr == bt:
    #         text = 'bottle'
    #         pass
    #     if pr == ct:
    #         text = 'cat'
    #         pass
    #     if pr == ch:
    #         text = 'chair'
    #         pass
    #     if pr == cp:
    #         text = 'computer'
    #         pass
    #     if pr == ctn:
    #         text = 'container'
    #         pass
    #     if pr == dg:
    #         text = 'dog'
    #         pass
    #     if pr == dr:
    #         text = 'door'
    #         pass
    #     if pr == fd:
    #         text = 'food'
    #         pass
    #     if pr == gl:
    #         text = 'glasses'
    #         pass
    #     if pr == ht:
    #         text = 'hat'
    #         pass
    #     if pr == ot:
    #         text = 'outlet'
    #         pass
    #     if pr == pn:
    #         text = 'pen'
    #         pass
    #     if pr == ph:
    #         text = 'phone'
    #         pass
    #     if pr == pl:
    #         text = 'pole'
    #         pass
    #     if pr == sf:
    #         text = 'sofa'
    #         pass
    #     if pr == tb:
    #         text = 'table'
    #         pass
    #     if pr == t:
    #         text = 'trash'
    #         pass
    #     if pr == tc:
    #         text = 'trash can'
    #         pass
    #     if pr == tr:
    #         text = 'tree'
    #         pass
    #     if pr == tv:
    #         text = 'tv'
    #         pass
    #     if pr == vl:
    #         text = 'vehicle'
    #         pass
    #     if pr == ws:
    #         text = 'wall switch'
    #         pass
    #     if pr == wh:
    #         text = 'watch'
    #         pass
    #     if pr == wn:
    #         text = 'window'
    #         pass

        # time.sleep(1)


    # for on, off, none, back in prediction:
    #     if float(on)>float(off) and float(on)>float(none) and float(on)>float(back):
    #         text = "mask"
    #         this = "Welcome to KJC"
    #     elif float(off)>float(on) and float(off)>float(none) and float(off)>float(back):
    #         text = "no mask"
    #         this = "Please Wear your mask"
    #     elif float(none)>float(on) and float(none)>float(off) and float(none)>float(back):
    #         text = "empty"
    #         this = ""
    #         pass
    #     elif float(back)>float(on) and float(back)>float(off) and float(back)>float(none):
    #         text = 'out'
    #         this = "Thanks for KJC"
    #         pass


        # print(text)
        
        # img = cv2.resize(img, (500, 500))
    # cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2)
        
    
    cv2.imshow('KJC Mask Scanner', frame)
    exitKey = cv2.waitKey(1)
    if exitKey == ord('q') or exitKey == ord('Q'):
        break

    # time.sleep(2)

cam.release()
cv2.destroyAllWindows()

