import numpy as np
import pickle
import cv2
import glob
import time

import matplotlib.image as mpi
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip as vfc
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as stsc
from sklearn.svm import LinearSVC

def set_globals():
    global svc, X_scaler, boxes, s_counter
    svc = pickle.load(open("svc.p", "rb"))
    X_scaler = pickle.load(open("X_scaler.p", "rb"))
    boxes = []
    s_counter = 0

def process_video(video_path, file_out):
    set_globals()
    output = file_out
    clip1 = vfc(video_path)#.subclip(t_start=25, t_end=35)
    clip = clip1.fl_image(process_frame)
    clip.write_videofile(output, audio=False)

def process_frame(img):
    global s_counter
    s_window = 20
    draw_img, bbox_list = find_cars(img, svc=svc, X_scaler=X_scaler, orient=9, pix_per_cell=8, cell_per_block=2,spatial_size=(16, 16), hist_bins=16, scales=(1.3,2.1))
    boxes.append(bbox_list)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    if s_counter > s_window:
        for i in range(0, s_window):
            heat = add_heat(heat, boxes[-i])
        heat = apply_threshold(heat, 1*s_window)
    else:
        heat = add_heat(heat, bbox_list)
        heat = apply_threshold(heat, 1)
    s_counter += 1
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

def get_hog_features(img, orient, pix_per_cell, cell_per_block,vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient,pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block),transform_sqrt=False,visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block),transform_sqrt=False,visualise=vis, feature_vector=feature_vec)
        return features

def bin_spat(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features

def c_hist(img, nbins=32, bins_range=(0, 256)):
    c1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    c2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    c3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((c1_hist[0], c2_hist[0], c3_hist[0]))
    return hist_features

def convert_colour(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def extract_features(imgs, cspace='RGB', orient=9,pix_per_cell=8, cell_per_block=2, hog_channel=0,spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    features = []
    for file in imgs:
        image = mpi.imread(file)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],orient, pix_per_cell, cell_per_block))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,pix_per_cell, cell_per_block)

        spatial_features = bin_spat(feature_image, size=spatial_size)
        hist_features = c_hist(
            feature_image, nbins=hist_bins, bins_range=hist_range)
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    return features

def load_images(folder_path='./data'):
    images_nv = glob.glob(folder_path + '/non-vehicles/non-vehicles/**/*.png')
    images_v = glob.glob(folder_path + '/vehicles/vehicles/**/*.png')
    cars = []
    notcars = []
    for image in images_nv:
        notcars.append(image)
    for image in images_v:
        cars.append(image)
    return cars, notcars

def train_test_svm(car_features, notcar_features):
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = stsc().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Feature vector length:', len(X_train[0]))
    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc, X_scaler

def extract_and_train(colorspace='RGB', orient=9, pix_per_cell=8,cell_per_block=2, hog_channel=0, spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    cars, notcars = load_images()
    car_features = extract_features(cars, cspace=colorspace, orient=orient,pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,hog_channel=hog_channel,spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)
    notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,hog_channel=hog_channel,spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)
    print('Using:', orient, 'orientations', pix_per_cell,'pixels per cell and', cell_per_block, 'cells per block')
    train_test_svm(car_features, notcar_features)

def find_cars(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, ystart=400, ystop=656, scales=(1.3,2.1)):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_colour(img_tosearch, conv='RGB2YCrCb')
    bbox_list = []
    for scale in scales:
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(
                imshape[1] / scale), np.int(imshape[0] / scale)))
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1
        nfeat_per_block = orient * cell_per_block**2
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        hog1 = get_hog_features(ch1, orient, pix_per_cell,cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell,cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell,cell_per_block, feature_vec=False)
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window,xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window,xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window,xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
                spatial_features = bin_spat(subimg, size=spatial_size)
                hist_features = c_hist(subimg, nbins=hist_bins)
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                    bbox_list.append(((xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return draw_img, bbox_list

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img

def main():
    process_video('./project_video.mp4', file_out='project_out_1.3_2.1.mp4')
    #print("Complete")
    return 0

if __name__ == "__main__":
    main()
