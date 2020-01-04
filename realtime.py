import numpy as np
import cv2
from heuristic_faces import HeuristicFaceClassifier
import pickle
import pandas as pd

cap = cv2.VideoCapture(0)
clf = HeuristicFaceClassifier()
f = open("horizontal_gaze.pkcls", "rb")

model = pickle.load(f)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = clf.detect_faces(frame)
    # Display the resulting frame
    for face in faces:
        (x, y, w, h) = face["face"]
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 0), 2)
        for eye in face["eyes"]:
            (ex, ey, ew, eh) = eye["eye"]
            ex, ey = x + ex, y + ey
            cv2.rectangle(gray, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
            pupil = eye["pupil"]
            cv2.circle(gray, (ex + pupil[0], ey + pupil[1]), 7, (255, 0, 0), 1)

        face_size = face['face'][2]
        dataframe = pd.DataFrame({
            'l_eye_x': face['eyes'][0]['eye'][0] / face_size,
            'l_eye_s': face['eyes'][0]['eye'][2] / face_size,
            'l_eye_px': face['eyes'][0]['pupil'][0] / face_size,
            'r_eye_x': face['eyes'][1]['eye'][0] / face_size,
            'r_eye_s': face['eyes'][1]['eye'][2] / face_size,
            'r_eye_px': face['eyes'][1]['pupil'][0] / face_size}, index=[0])

        prediction = model.predict(dataframe)[0]
        cv2.putText(gray, "Direction : " + str(prediction), (x, y), thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0))
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

a = input("Test")