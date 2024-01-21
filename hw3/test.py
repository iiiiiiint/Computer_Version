import argparse
import numpy as np
from pre_progress import FaceLib, Face
import cv2

FACE_PER_PERSON = 7
PERSON_NUM = 40
parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str)
parser.add_argument('filename', type=str)
args = parser.parse_args()

samples, faces = FaceLib(PERSON_NUM*FACE_PER_PERSON, PERSON_NUM, FACE_PER_PERSON, "./source")

model = np.load(args.filename)
e_vector_mat = model["e_vector_mat"]
e_value_mat = model["e_value_mat"]

distance = np.dot(e_vector_mat, samples)

# cnt = 0
#
# for i in range(1, PERSON_NUM + 1):
#     for j in range(FACE_PER_PERSON + 1, 11):
#         face_vect, origin_mat = Face(f"./source/s{i}/{j}")
#         face_vect = np.dot(e_vector_mat, face_vect)
#         distances = np.linalg.norm(face_vect - distance, axis=0, ord=2)
#         min_i = np.argmin(distances)
#
#         # Display result
#         similar_mat = faces[min_i]
#         text = f"s{(min_i // FACE_PER_PERSON) + 1} No.{(min_i % FACE_PER_PERSON) + 1}"
#         judge = bool()
#         if(i == (min_i // FACE_PER_PERSON) + 1):
#             cnt += 1
#             judge = True
#         else:
#             judge = False
#         print(text + " {}".format(judge))
#         cv2.putText(origin_mat, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2, 8)
#
# print("The accuracy is {:.4f}".format(cnt/((10 - FACE_PER_PERSON) * PERSON_NUM)))

face_vect, origin_mat = Face(args.image_path)
face_vect = np.dot(e_vector_mat, face_vect)
distances = np.linalg.norm(face_vect - distance, axis=0, ord=2)
min_i = np.argmin(distances)

similar_mat = faces[min_i]
text = f"s{(min_i // FACE_PER_PERSON) + 1} No.{(min_i % FACE_PER_PERSON) + 1}"
cv2.putText(origin_mat, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2, 8)

cv2.imshow("FaceResult", origin_mat)
cv2.imshow("Similar Pic", similar_mat)
cv2.waitKey(0)
cv2.destroyAllWindows()
