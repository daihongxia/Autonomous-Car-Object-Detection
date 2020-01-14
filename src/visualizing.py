import numpy as np
import cv2
from preprocessing import (get_img_coords, 
                            str2coords, 
                            rotate, 
                            coords_to_2d, 
                            preprocess_image, 
                            get_mask_and_regr,
                            camera_matrix,
                            camera_matrix_inv
                          )
from loader import imread
import matplotlib.pyplot as plt

# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    sin = np.sin
    cos = np.cos
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image

def visualize(img, coords):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    return img

def imshow_train(train, idx, PATH='../data/pku-autonomous-driving/'):
    plt.figure(figsize=(14,14))
    plt.imshow(imread(PATH + 'train_images/' + train['ImageId'][idx] + '.jpg'))
    plt.scatter(*get_img_coords(train['PredictionString'][idx]), color='red', s=100);

def show_img_mask(train_df, idx, PATH='../data/pku-autonomous-driving/'):
    img0 = imread(PATH + 'train_images/' + train_df['ImageId'][idx] + '.jpg')
    img = preprocess_image(img0)

    mask, regr, mask_gaus = get_mask_and_regr(img0, train_df['PredictionString'][idx])

    print('img.shape', img.shape, 'std:', np.std(img))
    print('mask.shape', mask.shape, 'std:', np.std(mask))
    print('regr.shape', regr.shape, 'std:', np.std(regr))

    plt.figure(figsize=(16,16))
    plt.title('Processed image')
    plt.imshow(img)
    plt.show()

    plt.figure(figsize=(16,16))
    plt.title('Detection Mask')
    plt.imshow(mask)
    plt.show()

    plt.figure(figsize=(16,16))
    plt.title('Detection Mask Gaussian')
    plt.imshow(mask_gaus)
    plt.show()

    plt.figure(figsize=(16,16))
    plt.title('Yaw values')
    plt.imshow(regr[:,:,-2])
    plt.show()