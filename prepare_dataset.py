import cv2
import os
import numpy as np
from heuristic_faces import HeuristicFaceClassifier
import pandas as pd

def run():
    heuristic_clf = HeuristicFaceClassifier()
    data_path = os.getcwd() + "\datasets\columbia_gaze_data_set"
    output_image_path = os.getcwd() + "\datasets\columbia_heuristic_gazes\images\\"
    output_data_path = os.getcwd() + "\datasets\columbia_heuristic_gazes\data\\"

    dot_position_file = open(output_data_path + 'dot_position.dat', 'a')

    dataframe = pd.DataFrame([], columns=['subject', 'l_eye_x', 'l_eye_y', 'l_eye_s', 'l_eye_px', 'l_eye_py', 'r_eye_x', 'r_eye_y', 'r_eye_s', 'r_eye_px', 'r_eye_py', 'g_v', 'g_h'])

    count = 0
    eye_w = []
    eye_h = []
    folders = os.listdir(data_path)
    for folder in folders:
        current_folder = data_path + "\\" + folder
        images = os.listdir(current_folder)
        dot_positions = []
        camera_h_positions = []
        camera_v_positions = []



        for image in images:
            if image != 'Thumbs.db' and image != '.picasa.ini' and image != '.DS_Store':

                img = cv2.imread(current_folder + "\\" + image)
                img = cv2.resize(img, None, fx=0.1, fy=0.1)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                heuristic_faces = heuristic_clf.detect_faces(img)

                for face in heuristic_faces:

                    (x, y, w, h) = face["face"]
                    crop = img_gray[y:y + h, x: x + w]
                    #
                    for eye in face["eyes"]:
                        (x, y, w, h) = eye['eye']
                        eye_h.append(h)
                        eye_w.append(w)
                        cv2.rectangle(crop, (x, y), (x + w, y + h), (255, 255, 0), 2)

                        pupil = eye["pupil"]
                        cv2.circle(crop, (x + pupil[0], y + pupil[1]), 7, (255, 0, 0), 2)

                    # print(face)
                    # cv2.imshow('crop', crop)
                    # cv2.imwrite('crop.jpg', crop)
                    # cv2.waitKey()

                    image_data = image.split(".")[0].split("_")
                    subject = int(image_data[0])
                    dot_position = int(image_data[2].replace("P", ""))
                    camera_v_position = int(image_data[3].replace("V", ""))
                    camera_h_position = int(image_data[4].replace("H", ""))
                    #
                    # cv2.imwrite(output_image_path + str(count) + '.jpg', crop)
                    # dot_position_file.write(str(dot_position) + "\n")
                    #

                    face_size = face['face'][2]
                    dataframe = dataframe.append({
                        'subject': subject,
                        'l_eye_x': face['eyes'][0]['eye'][0]/face_size,
                        'l_eye_y': face['eyes'][0]['eye'][1]/face_size,
                        'l_eye_s': face['eyes'][0]['eye'][2]/face_size,
                        'l_eye_px': face['eyes'][0]['pupil'][0]/face_size,
                        'l_eye_py': face['eyes'][0]['pupil'][1]/face_size,
                        'r_eye_x': face['eyes'][1]['eye'][0]/face_size,
                        'r_eye_y': face['eyes'][1]['eye'][1]/face_size,
                        'r_eye_s': face['eyes'][1]['eye'][2]/face_size,
                        'r_eye_px': face['eyes'][1]['pupil'][0]/face_size,
                        'r_eye_py': face['eyes'][1]['pupil'][1]/face_size,
                        'g_v': camera_v_position,
                        'g_h': dot_position - camera_h_position
                    }, ignore_index=True)

                    count = count + 1
                    print(count)

    # dot_position_file.close()
    dataframe.to_csv('data.csv')
    print(min(eye_h))
    print(min(eye_w))


run()
