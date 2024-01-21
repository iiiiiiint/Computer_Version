import argparse
import numpy as np
from pre_progress import FaceLib
import cv2


FACE_PER_PERSON = 7
PERSON_NUM = 40

parser = argparse.ArgumentParser()
parser.add_argument('percent', type=float)
parser.add_argument('filename', type=str)
args = parser.parse_args()

samples, _ = FaceLib(PERSON_NUM*FACE_PER_PERSON, PERSON_NUM, FACE_PER_PERSON, "./source")

print("Calculating Covariance Mat...")
cov_mat, mean_mat = np.cov(samples, rowvar=False), np.mean(samples, axis=0)
cov_mat /= (samples.shape[1] - 1)
samples = samples.astype(np.float64)
samples -= mean_mat
e_value_mat, e_vector_mat = np.linalg.eigh(cov_mat)
e_vector_mat = (samples @ e_vector_mat.T).T

value_sum = np.sum(e_value_mat)
energy_level = value_sum * args.percent

energy_sum, k = 0, 0
for k in range(e_value_mat.shape[0]):
    energy_sum += e_value_mat[k]
    if energy_sum >= energy_level:
        break

e_vector_mat = e_vector_mat[:k, :]
print(e_vector_mat.shape)
np.savez(args.filename, e_vector_mat=e_vector_mat, e_value_mat=e_value_mat[:k])
top_10_eigenfaces = [cv2.normalize(e_vector_mat[i].reshape(89, 73), None, 1.0, 0.0, cv2.NORM_MINMAX) for i in range(min(10, k))]
result = np.concatenate(top_10_eigenfaces, axis=1)
result = (result * 255).astype(np.uint8)

cv2.imshow("Top10EigenFace", result)
cv2.imwrite("Top10EigenFace.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
