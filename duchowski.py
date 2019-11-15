import cv2
import json
import numpy as np

def run():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    images = ["test1.jpg"
              # , "img2.JPG"
              ]
    for image in images:
        img = cv2.imread(image)
        img = cv2.resize(img, None, fx=0.1, fy=0.1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        newGray = gray[:]
        cv2.equalizeHist(newGray, gray)
        faces = face_cascade.detectMultiScale(gray)
        eyes = eye_cascade.detectMultiScale(gray)
        test(img, faces)
        cv2.imshow('img', img)
        cv2.imwrite('img.jpg', img)
        cv2.waitKey()

    # vidcap = cv2.VideoCapture('testvideo.mp4')
    #
    # frame_width = int(vidcap.get(3))
    # frame_height = int(vidcap.get(4))
    #
    # out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, (frame_width, frame_height))
    # i = 0
    # while vidcap.isOpened():
    #     success, img = vidcap.read()
    #     if success:
    #         if i % 5 == 0:
    #             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #             faces = face_cascade.detectMultiScale(gray)
    #             # eyes = eye_cascade.detectMultiScale(gray)
    #             test(img, faces)
    #         # out.write(img)
    #             print(i)
    #         i = i + 1
    #         # if i == 500:
    #         #     break
    #         cv2.imshow("img", img)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):                     # exit if Escape is hit
    #             break
    #     else:
    #         break
    #
    # vidcap.release()
    # out.release()

    cv2.destroyAllWindows()


def test(img, faces):
    i = 0
    for (x, y, w, h) in faces:
        crop = img[y:y + h, x: x + w]
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        eyes = eye_cascade.detectMultiScale(gray)

        # if len(eyes) == 0:
        #     continue

        # cv2.imwrite("output\\faces\\" + str(i) + ".jpg", gray)
        # face_info = {
        #     "x": float(x),
        #     "y": float(y),
        #     "w": float(w),
        #     "h": float(h)
        # }
        # file = open("output\\faces\\" + str(i) + ".json", "w")
        # print(type(face_info))
        # json.dump(face_info, file)
        # file.close()
        cv2.waitKey()

        # eye lines - vertical
        # cv2.line(img, (x + int(w / 5.0), y), (x + int(w / 5.0), y + h), (0, 255, 0), 2)
        # cv2.line(img, (x + int(4 * w / 5.0), y), (x + int(4 * w / 5.0), y + h), (0, 255, 0), 2)

        # eye lines - horizontal
        # cv2.line(img, (x, y + int(h / 3.0)), (x + w, y + int(h / 3.0)), (0, 255, 0), 2)

        # nose line - vertical
        # cv2.line(img, (x + int(w / 2.0), y), (x + int(w / 2.0), y + h), (0, 255, 0), 1)
        # cv2.line(img, (x + int(2 * w / 5.0), y), (x + int(2 * w / 5.0), y + h), (0, 255, 0), 1)
        # cv2.line(img, (x + int(3 * w / 5.0), y), (x + int(3 * w / 5.0), y + h), (0, 255, 0), 1)
        eyes = check_eyes_in_face((x, y, w, h), eyes)
        if len(eyes) != 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 255, 0), 2)

    # for (x, y, w, h) in eyes:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # # cv2.imwrite('img.jpg', img)
    # cv2.waitKey()


def check_eyes_in_face(face, eyes):
    (x, y, w, h) = face
    if len(eyes) ==2:
        # we have two eyes
        (e1x, e1y, e1w, e1h) = eyes[0]
        (e2x, e2y, e2w, e2h) = eyes[1]

        # check eyes same side of 1/5 , 4/5 line
        # eye should be same side of 1/5 line
        (x1, y1) = (x + int(w / 5.0), y)
        (x2, y2) = (x + int(w / 5.0), y + h)
        d1 = ((e1x - x1) * (y2 - y1)) - ((e1y - y1) * (x2 - x1))
        d2 = ((e2x - x1) * (y2 - y1)) - ((e2y - y1) * (x2 - x1))
        if (d1 > 0) ^ (d2 > 0):
            return []

        (x1, y1) = (x + int(w * 4 / 5.0), y)
        (x2, y2) = (x + int(w * 4 / 5.0), y + h)
        d1 = ((e1x - x1) * (y2 - y1)) - ((e1y - y1) * (x2 - x1))
        d2 = ((e2x - x1) * (y2 - y1)) - ((e2y - y1) * (x2 - x1))
        if (d1 > 0) ^ (d2 > 0):
            return []
        # check eyes either side of mid line
        (x1, y1) = (x + int(w / 2.0), y)
        (x2, y2) = (x + int(w / 2.0), y + h)
        d1 = ((e1x - x1) * (y2 - y1)) - ((e1y - y1) * (x2 - x1))
        d2 = ((e2x - x1) * (y2 - y1)) - ((e2y - y1) * (x2 - x1))
        if not ((d1 > 0) ^ (d2 > 0)):
            return eyes

        return True
    elif len(eyes) == 1:
        # Only one eye detected
        (e1x, e1y, e1w, e1h) = eyes[0]
        # check side of face
        (x1, y1) = (x + int(w / 2.0), y)
        (x2, y2) = (x + int(w / 2.0), y + h)
        d1 = ((e1x - x1) * (y2 - y1)) - ((e1y - y1) * (x2 - x1))
        if d1 < 0:
            eyes = np.append(eyes, [[e1x + 5, e1y + 5, e1w, e1h]], axis=0)
        else:
            eyes = np.append(eyes, [[e1x, e1y, e1w, e1h]], axis=0)
        return eyes
    return []
