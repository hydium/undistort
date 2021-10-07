import numpy as np
import cv2 as cv
import glob

# def correct_pt(uv_3, K, Kinv, ds):
#    # uv_3 = np.stack((uv[:,0],uv[:,1],np.ones(uv.shape[0]),),axis=-1)
#    xy_= np.matmul(Kinv, uv_3)
#    r = np.linalg.norm(xy_,axis=-1)
#    # coeff=(1 + ds[0] * (r ** 2) + ds[1] * (r ** 4) + ds[4] * (r ** 6));
#    # xy__ = xy_*coeff[:,np.newaxis]

#    xy_[0] = (1 + ds[0] * (r ** 2) + ds[1] * (r ** 4) + ds[4] * (r ** 6)) * xy_[0] + 2 * ds[2] * xy_[0] * xy_[1] + ds[3] * (r ** 2 + 2 * xy_[0] ** 2)

#    xy_[1] = (1 + ds[0] * (r ** 2) + ds[1] * (r ** 4) + ds[4] * (r ** 6)) * xy_[1] + 2 * ds[3] * xy_[0] * xy_[1] + ds[2] * (r ** 2 + 2 * xy_[1] ** 2)

#    return np.matmul(K, xy_)[:2]


def undistort(xy, k, distortion, iter_num=5):
    k1, k2, p1, p2, k3 = distortion
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    x, y = xy.astype(float)
    x = (x - cx) / fx
    x0 = x
    y = (y - cy) / fy
    y0 = y
    for _ in range(iter_num):
        r2 = x ** 2 + y ** 2
        k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
        delta_x = 2 * p1 * x*y + p2 * (r2 + 2 * x**2)
        delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x*y
        x = (x0 - delta_x) * k_inv
        y = (y0 - delta_y) * k_inv

        # x = (x0 + delta_x) / k_inv
        # y = (y0 + delta_y) / k_inv

    return np.array((x * fx + cx, y * fy + cy))


k = np.array([[753.349186340502,	0,	1010.31833065182],
			 [0, 753.143587767122,	588.647123579411],
			 [0, 0,	1]])

distortion = np.array([-0.358074811139381, 0.150366096279157, -0.000239617440106,	-0.001364488806427,	-0.031502910462795])

# Kinv = np.linalg.inv(K)

img = cv.imread('image_000100.jpg')

h,  w = img.shape[:2]

print(img.shape)

# print(Kinv)

cv.imwrite("test1.png", img)

corrected_img = np.zeros(img.shape)

# print(np.array([float(1) / 1080.0 * 2 - 1, float(1) / 1920.0 * 2 - 1, 1]))

test = np.array([h, w])

print(test)

for i in range(h):
	for j in range(w):
		corrected_point = undistort(np.array([i - 540, j - 960]), k, distortion)

		# print(corrected_point)

		if (abs(corrected_point[0]) > h or abs(corrected_point[1]) > w):
			continue;

		corrected_img[int(corrected_point[0])][int(corrected_point[1])] = img[i][j]

cv.imwrite("test.png", corrected_img)

# newimg = np.zeros(img.shape

# for i 