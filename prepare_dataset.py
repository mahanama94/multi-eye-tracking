import cv2
import os
import numpy as np
from heuristic_faces import HeuristicFaceClassifier


def run():
    heuristic_clf = HeuristicFaceClassifier()
    data_path = os.getcwd() + "\datasets\columbia_gaze_data_set"
    output_image_path = os.getcwd() + "\datasets\columbia_heuristic_gazes\images\\"
    output_data_path = os.getcwd() + "\datasets\columbia_heuristic_gazes\data\\"

    dot_position_file = open(output_data_path + 'dot_position.dat', 'a')

    count = 0
    folders = os.listdir(data_path)
    for folder in folders:
        current_folder = data_path + "\\" + folder
        images = os.listdir(current_folder)
        dot_positions = []
        camera_h_positions = []
        camera_v_positions = []

        for image in images:
            if image != 'Thumbs.db' and image != '.picasa.ini' and image != '.DS_Store':

                print(image)

                img = cv2.imread(current_folder + "\\" + image)
                img = cv2.resize(img, None, fx=0.1, fy=0.1)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                heuristic_faces = heuristic_clf.detect_faces(img)

                if len(heuristic_faces) != 0:
                    (x, y, w, h) = heuristic_faces[0]
                    crop = img_gray[y:y + h, x: x + w]
                    #
                    # cv2.imshow('crop', crop)
                    # cv2.waitKey()

                    image_data = image.split(".")[0].split("_")
                    subject = int(image_data[0])
                    dot_position = int(image_data[2].replace("P", ""))
                    camera_v_position = int(image_data[3].replace("V", ""))
                    camera_h_position = int(image_data[4].replace("H", ""))

                    cv2.imwrite(output_image_path + str(count) + '.jpg', crop)
                    dot_position_file.write(str(dot_position) + "\n")

                    count = count + 1


                # print(image_data)

    dot_position_file.close()


run()
