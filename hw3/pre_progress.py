import os
import cv2
import numpy as np


def Face(path):
    with open(path + ".txt", "r") as file:
        line = file.readline()
        number = line.split()
    x1, y1, x2, y2 = float(number[0]), float(number[1]), float(number[2]), float(number[3])
    origin_image = cv2.imread(path + ".pgm", cv2.IMREAD_COLOR)
    grey_image = cv2.imread(path + ".pgm", cv2.IMREAD_GRAYSCALE)
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
    trans_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    trans_mat[0, 2] += 37 - center[0]
    trans_mat[1, 2] += 30 - center[1]
    transformed_pic = cv2.warpAffine(grey_image, trans_mat,
                                          (int(grey_image.shape[1] * 0.8), int(grey_image.shape[0] * 0.8)),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0.0, 0.0, 0.0))
    transformed_pic = cv2.equalizeHist(transformed_pic)
    equalized_mat = transformed_pic.copy()
    vect = equalized_mat.reshape(1, -1).T
    return vect, origin_image


def FaceLib(face_num, person_num, face_per_person, path):
    faces = []
    vects = []
    for i in range(1, person_num + 1):
        for j in range(1, face_per_person + 1):
            path_ = "{}/s{}/{}".format(path, i, j)
            vect, origin_image = Face(path_)
            faces.append(origin_image)
            vects.append(vect)
    samples = cv2.hconcat(vects)
    return samples, faces
