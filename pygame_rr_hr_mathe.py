import pygame
from pygame.locals import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy import interpolate
from scipy import stats
from scipy import signal
import pylab as pl
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from PIL import Image


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cfx_face = np.array([])
cfy_face = np.array([])
cfx_chest = np.array([])
cfy_chest = np.array([])

cfx_teliko_array = np.array([])
cfy_teliko_array = np.array([])

cfx_teliko_array_chest = np.array([])
cfy_teliko_array_chest = np.array([])

WHITE = (255, 255, 255)

color = np.random.randint(0, 255, (10000, 3))
cap = cv2.VideoCapture(0)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def initial():
    global corners, x, y, h, w, x_new, y_new, h_new, w_new
    # open video capture

    # create some random colors for lukas-kanade visualization
    # megalh eikona
    # take first frame and find corners in it
    ret, first_frame = cap.read()

    # parameters for lucas kanade optical flow
    # our lk#lk_params = dict( winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    # megalh gkri eiko
    # create a gray copy of first frame
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # psaxno prosopo sth megalh gri
    # apply face detector

    faces = face_cascade.detectMultiScale(first_frame_gray, 1.3, 5)
    chest = face_cascade.detectMultiScale(first_frame_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # create rectangle of detected face
        # cv2.rectangle(first_frame,(x,y),(x+w,y+h),(255,0,0),2)

        # 2 mikres eikones

        # select roi of face
        roi_first_gray = first_frame_gray[y:y + h, x:x + w]
        # select color roi of face
        roi_first_color = first_frame[y:y + h, x:x + w]

        # heuristic parameters for facial features
        x_new = int(0.25 * w)
        y_new = int(0.05 * h)
        h_new = int(0.9 * h)
        w_new = int(0.5 * w)

        # create rectangle using heuristic parameters
        roi_first_gray_face = first_frame_gray[y:y + h, x:x + w]
        roi_first_color_face = first_frame[y:y + h, x:x + w]

        # extract good features to track from gray roi of face
        # our corners:#corners = cv2.goodFeaturesToTrack(roi_first_gray,1000,0.003,3)
        corners = cv2.goodFeaturesToTrack(roi_first_gray,500,0.0125,1, useHarrisDetector=0, k=0.04)
        corners2 = []

        j = 0

        p0 = np.int0(corners)
        for kk in p0:
            x1, y1 = kk.ravel()
            if x1 > x_new and x1 < x_new + w_new and (
                    (y1 > y_new and y1 < y_new + 0.2 * h_new) or (y1 > y_new + 0.55 * h_new and y1 < y_new + h_new)):
                temp = np.empty((1, 2))
                temp[0][0] = x1
                temp[0][1] = y1
                corners2.append(temp)
                j = j + 1
                cv2.circle(roi_first_color_face, (x1, y1), 3, 255, -1)

    corners_face_arr = np.float32(np.array(corners2))

    for (x, y, w, h) in chest:
        # create rectangle of detected face
        # cv2.rectangle(first_frame,(x,y),(x+w,y+h),(255,0,0),2)

        # 2 mikres eikones

        # select roi of face
        roi_first_gray = first_frame_gray[y:y + h, x:x + w]
        # select color roi of face
        roi_first_color = first_frame[y:y + h, x:x + w]

        # heuristic parameters for facial features
        x_new = int(x - 0.5 * w)
        y_new = int(y + 1.3 * h)
        h_new = int(1 * h)
        w_new = int(1.7 * w)
ς
        # create rectangle using heuristic parameters
        roi_first_gray_chest = first_frame_gray[y:y_new + h_new, x:x_new + w_new]
        roi_first_color_chest = first_frame[y:y_new + h_new, x:x_new + w_new]

        # extract good features to track from gray roi of face
        # our corners:#corners = cv2.goodFeaturesToTrack(roi_first_gray,1000,0.003,3)
        corners = cv2.goodFeaturesToTrack(roi_first_gray_chest, 500, 0.005, 3)
        corners2_chest = []

        j = 0

        p0 = np.int0(corners)
        for kk in p0:
            x2, y2 = kk.ravel()
            if x2 > x_new and x2 < x_new + w_new and (
                    (y2 > y_new and y2 < y_new + h_new) or (y2 > y_new + 0.55 * h_new and y2 < y_new + h_new)):
                temp = np.empty((1, 2))
                temp[0][0] = x2
                temp[0][1] = y2
                corners2_chest.append(temp)
                j = j + 1
                cv2.circle(roi_first_color_chest, (x2, y2), 3, 255, -1)

    corners_chest_arr = np.float32(np.array(corners2_chest))

    mask_face = np.zeros_like(roi_first_color)
    mask_chest = np.zeros_like(roi_first_color_chest)

    return faces,chest,first_frame,first_frame_gray,lk_params,corners_face_arr,corners_chest_arr,roi_first_gray_chest,roi_first_color_chest,roi_first_gray_face,roi_first_color_face,mask_face,mask_chest

def tracking(roi_first_gray_face,corners2_face_arr,lk_params,roi_first_gray_chest,corners2_arr_chest,mask_face,mask_chest, counter_while, cfx_face,cfy_face,cfx_chest,cfy_chest, point):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_frame_gray_face = frame_gray[y:y + h, x:x + w]
    roi_frame_color_face = frame[y:y + h, x:x + w]

    roi_frame_gray_chest = frame_gray[y:y_new + h_new, x:x_new + w_new]
    roi_frame_color_chest = frame[y:y_new + h_new, x:x_new + w_new]

    if point== True:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 255, 0), 2)
    # gianampei edw to roi_first prepei kai to diplano toy na einai idio megethos
    p1, st, err = cv2.calcOpticalFlowPyrLK(roi_first_gray_face, roi_frame_gray_face, corners2_face_arr, None, **lk_params)

    p1_chest, st_chest, err_chest = cv2.calcOpticalFlowPyrLK(roi_first_gray_chest, roi_frame_gray_chest, corners2_arr_chest, None, **lk_params)

    # Select good points /// to p1 exei x kai y kai prosdiorizei ta kainouria
    # to corners2_arr einai ta shmeia poy exoume vrei prwta
    good_new_face = p1[st == 1]
    good_old_face = corners2_face_arr[st == 1]

    good_new_chest = p1_chest[st_chest == 1]
    good_old_chest = corners2_arr_chest[st_chest == 1]

    # dhlwsh-arxikopoihsh pinakwn
    cfx_for_each_frame_face = np.array([])
    cfy_for_each_frame_face = np.array([])

    cfx_for_each_frame_chest = np.array([])
    cfy_for_each_frame_chest = np.array([])

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new_face, good_old_face)):
        a, b = new.ravel()
        c, d = old.ravel()

        cfx_for_each_frame_face = np.append(cfx_for_each_frame_face, a)
        cfy_for_each_frame_face = np.append(cfy_for_each_frame_face, b)

        cv2.line(mask_face, (a, b), (c, d), color[i].tolist(), 1)
        cv2.circle(roi_frame_color_face, (a, b), 2, color[i].tolist(), -1)

    for i, (new_chest, old_chest) in enumerate(zip(good_new_chest, good_old_chest)):
        a, b = new_chest.ravel()
        c, d = old_chest.ravel()

        cfx_for_each_frame_chest = np.append(cfx_for_each_frame_chest, a)
        cfy_for_each_frame_chest = np.append(cfy_for_each_frame_chest, b)

        cv2.line(mask_chest, (a, b), (c, d), color[i].tolist(), 1)
        cv2.circle(roi_frame_color_chest, (a, b), 2, color[i].tolist(), -1)

    # kanw reshape gia na einai idia me to cfx kai y
    cfx_for_each_frame_face = cfx_for_each_frame_face.reshape((-1, 1))
    cfy_for_each_frame_face = cfy_for_each_frame_face.reshape((-1, 1))

    cfx_for_each_frame_chest = cfx_for_each_frame_chest.reshape((-1, 1))
    cfy_for_each_frame_chest = cfy_for_each_frame_chest.reshape((-1, 1))

    if counter_while == 0:
        cfx_face = cfx_for_each_frame_face
        cfy_face = cfy_for_each_frame_face

    if counter_while == 0:
        cfx_chest = cfx_for_each_frame_chest
        cfy_chest = cfy_for_each_frame_chest

    if counter_while != 0:
        cfx_face = np.concatenate((cfx_face, cfx_for_each_frame_face), axis=1)
        cfy_face = np.concatenate((cfy_face, cfy_for_each_frame_face), axis=1)

    if counter_while != 0:
        cfx_chest = np.concatenate((cfx_chest, cfx_for_each_frame_chest), axis=1)
        cfy_chest = np.concatenate((cfy_chest, cfy_for_each_frame_chest), axis=1)

    img_face = cv2.add(roi_frame_color_face, mask_face)
    img_chest = cv2.add(roi_frame_color_chest, mask_chest)

    #cv2.imshow('frame5', img_chest)
    k_chest = cv2.waitKey(30) & 0xff

    #cv2.imshow('frame', img_face)
    k_face = cv2.waitKey(30) & 0xff

    roi_first_gray_face= roi_frame_gray_face.copy()
    corners2_arr_face = good_new_face.reshape(-1, 1, 2)

    roi_first_gray_chest = roi_frame_gray_chest.copy()
    corners2_arr_chest = good_new_chest.reshape(-1, 1, 2)

    #cv2.imshow('frame4', frame)

    return corners2_arr_face,corners2_arr_chest,cfx_face,cfy_face,cfx_chest,cfy_chest,frame, roi_first_gray_face, roi_first_gray_chest, frame

def cal_cf_face(cfx_face, cfy_face, cfx_teliko_array, cfy_teliko_array, frames_1, start):
    end = time.time()


    round_array1 = np.array([])
    kinhsh_array1 = np.array([])
    max_distance_array1 = np.array([])
    cfx_teliko = np.array([])

    print(end-start)

    interpol= 5

    for i in range(0, cfx_face.shape[0]):
        x_axis1 = np.arange(0, frames_1, 1)
        y_axis1 = np.array([])
        interpolation_x = np.array([])
        max_distance1 = np.array([])
        kinhsh_temp_array1 = np.array([])

        for j in range(0, counter_while - 1):
            # sto y_axis kanw append- bazw thn kathe grammh
            y_axis1 = np.append(y_axis1, cfx_face[i, j])
        # to y_axis exei gemisei me thn prwth grammh kanw reshape
        # kanw cubic interpolation se kathe grammh kai reshape sta new x kai y
        tck1 = interpolate.splrep(x_axis1, y_axis1, k=3, s=2)
        x_axis_new1 = np.arange(0, frames_1, 0.125)
        y_axis_new1 = interpolate.splev(x_axis_new1, tck1, der=0)

        if i == 0:
            interpolation_array1 = np.zeros(y_axis_new1.shape)
            interpolation_array1 = interpolation_array1.reshape((-1, 1))
            kinhsh_array1 = np.zeros(counter_while - 1)
            kinhsh_array1 = kinhsh_array1.reshape((-1, 1))
        # sto interpolation_y thelw na kanw append tis times twn y gia kathe grammi ta opoia exoun ginei perisotera logo ts interpolation
        interpolation_x = np.append(interpolation_x, y_axis_new1)
        interpolation_x = np.matrix(interpolation_x)
        interpolation_x = interpolation_x.T
        # ston interpolation pinaka thelw na ftiaksw ton teliko pinaka NxN me ta stoixeia logo ths interpolation
        interpolation_array1 = np.concatenate((interpolation_array1, interpolation_x), axis=1)

        for j in range(0, counter_while - 1):
            kinhsh1 = np.fabs(cfx_face[i, j] - cfx_face[i, j + 1])
            # print("ayto to stoixeio einai to [ ",i,",",j,"] kai h diafora metaksi n me to nk+1 frame==",kinhsh)
            kinhsh_temp_array1 = np.append(kinhsh_temp_array1, kinhsh1)
        # print("array kinhsh",kinhsh_temp_array)
        # print("shape",kinhsh_temp_array.shape)
        # exeis shape (1,)...(5,)

        # to max distance tha exei to megisto kai o array tha ta exei ola
        max_distance1 = np.amax(kinhsh_temp_array1)
        max_distance_array1 = np.append(max_distance_array1, max_distance1)
        # o round tha ta exei stoggulopoihmena
        round_array1 = np.around(max_distance_array1, decimals=1)
        # o pinakas kinhsh array tha exei to synoliko pinaka metakinhsewn
        kinhsh_temp_array1 = kinhsh_temp_array1.reshape((-1, 1))
        kinhsh_array1 = np.concatenate((kinhsh_array1, kinhsh_temp_array1), axis=1)

    interpolation_array1 = interpolation_array1.T
    timh_katanomhs1 = stats.mode(round_array1)
    cfx_teliko_array = np.array([])
    # h where returns ndarray or tuple of ndarrays
    cfx_teliko = np.where(round_array1 <= timh_katanomhs1[0])
    # print("to cfx teliko einai toy where:",cfx_teliko[0])
    # print("to cfy teliko len einai:",len(cfy_teliko))
    # to cfy_teliko exei tiw theseis twn stoixeiwn toy y gia to cfx poy xreiazomatse
    cfx_teliko_array = np.append(cfx_teliko_array, cfx_teliko[0])
    # print("cfx_teliko_array  theseis twn stoixeiwn toy y gia to cfx poy xreiazomatse",cfx_teliko_array.shape[0])
    # ------------------------------------------------------------------------------------swsta---------------------
    final21 = np.array([])

    interpolation_array1 = interpolation_array1.astype(int)
    cfx_teliko_array = cfx_teliko_array.astype(int)

    for i in range(0, cfx_teliko_array.shape[0]):  # apo 0 ews ta telika y poy thelw
        final1 = np.array([])
        final1 = np.append(final1, interpolation_array1[cfx_teliko_array[i]])
        final21 = np.concatenate((final21, final1), axis=0)

    final21 = np.reshape(final21, (cfx_teliko_array.shape[0], -1))
    # ---------- mean kai covariance-----------
    final_mean1 = np.array([])
    final_mean_value1 = np.array([])
    for i in range(0, final21.shape[1]):
        mean_value1 = np.array([])
        mean_value1 = np.mean(final21, axis=0)

    covariance_x = np.cov(final21)
    # telos tou x---------------------------------------------------------------------------------------------------------
    round_array = np.array([])
    kinhsh_array = np.array([])
    max_distance_array = np.array([])
    cfy_teliko = np.array([])
    for i in range(0, cfy_face.shape[0]):
        x_axis = np.arange(0, frames_1, 1)
        y_axis = np.array([])
        interpolation_y = np.array([])
        max_distance = np.array([])
        kinhsh_temp_array = np.array([])

        for j in range(0, counter_while - 1):
            # sto y_axis kanw append- bazw thn kathe grammh
            y_axis = np.append(y_axis, cfy_face[i, j])
        # to y_axis exei gemisei me thn prwth grammh kanw reshape
        # kanw cubic interpolation se kathe grammh kai reshape sta new x kai y
        tck = interpolate.splrep(x_axis, y_axis, s=0)
        x_axis_new = np.arange(0, frames_1, 1 / interpol)
        y_axis_new = interpolate.splev(x_axis_new, tck, der=0)

        if i == 0:
            interpolation_array = np.zeros(y_axis_new.shape)
            interpolation_array = interpolation_array.reshape((-1, 1))

            kinhsh_array = np.zeros(counter_while - 1)
            kinhsh_array = kinhsh_array.reshape((-1, 1))
        # sto interpolation_y thelw na kanw append tis times twn y gia kathe grammi ta opoia exoun ginei perisotera logo ts interpolation
        interpolation_y = np.append(interpolation_y, y_axis_new)
        interpolation_y = np.matrix(interpolation_y)
        interpolation_y = interpolation_y.T
        # ston interpolation pinaka thelw na ftiaksw ton teliko pinaka NxN me ta stoixeia logo ths interpolation
        interpolation_array = np.concatenate((interpolation_array, interpolation_y), axis=1)

        for j in range(0, counter_while - 1):
            kinhsh = np.fabs(cfy_face[i, j] - cfy_face[i, j + 1])
            # print("ayto to stoixeio einai to [ ",i,",",j,"] kai h diafora metaksi n me to nk+1 frame==",kinhsh)
            kinhsh_temp_array = np.append(kinhsh_temp_array, kinhsh)
        # exeis shape (1,)...(5,)
        # to max distance tha exei to megisto kai o array tha ta exei ola
        max_distance = np.amax(kinhsh_temp_array)
        max_distance_array = np.append(max_distance_array, max_distance)
        # o round tha ta exei stoggulopoihmena
        round_array = np.around(max_distance_array, decimals=1)
        # o pinakas kinhsh array tha exei to synoliko pinaka metakinhsewn
        kinhsh_temp_array = kinhsh_temp_array.reshape((-1, 1))
        kinhsh_array = np.concatenate((kinhsh_array, kinhsh_temp_array), axis=1)

    interpolation_array = interpolation_array.T
    timh_katanomhs = stats.mode(round_array)
    cfy_teliko_array = np.array([])
    # h where returns ndarray or tuple of ndarrays
    cfy_teliko = np.where(round_array <= timh_katanomhs[0])
    # print("to cfy teliko einai toy where:",cfy_teliko[0])
    # print("to cfy teliko len einai:",len(cfy_teliko))
    # to cfy_teliko exei tiw theseis twn stoixeiwn toy y gia to cfx poy xreiazomatse
    cfy_teliko_array = np.append(cfy_teliko_array, cfy_teliko[0])
    # print("cfy_teliko_array  theseis twn stoixeiwn toy y gia to cfx poy xreiazomatse",cfy_teliko_array.shape[0])

    # ------------------------------------------------------------------------------------swsta---------------------
    final2 = np.array([])

    interpolation_array = interpolation_array.astype(int)
    cfy_teliko_array = cfy_teliko_array.astype(int)

    for i in range(0, cfy_teliko_array.shape[0]):  # apo 0 ews ta telika y poy thelw

        final = np.array([])
        final = np.append(final, interpolation_array[cfy_teliko_array[i]])
        final2 = np.concatenate((final2, final), axis=0)

    final2 = np.reshape(final2, (cfy_teliko_array.shape[0], -1))

    # print(final2[5].shape)
    x1 = range(final2[5].shape[0])
    # print(x1)

    pl.plot(x1, final2[5])
    # pl.plot(x1, final2[1],'o')
    plt.title('y-position of a feature after interpolation')
    plt.xlabel('sample number')
    plt.ylabel('y-position')
    plt.grid(True)

    # pl.show()
    # final2: the y-trajectories of the interpolation

    fs = ((1 + interpol) * frames_1) / (end - start)
    lowcut = 0.8
    highcut = 2

    # print("final2 old", final2.shape)
    # print("iter=",final2.shape[0])
    final2but = np.array([])

    for m in range(0, final2.shape[0]):
        but = np.array([])
        but = butter_bandpass_filter(final2[m, :], lowcut, highcut, fs, 5)
        #	print("but size=", but.shape)

        final2but = np.concatenate((final2but, but), axis=0)

    final2but = np.reshape(final2but, (final2.shape[0], -1))
    final2 = final2but

    x1 = range(final2[5].shape[0])
    # print(x1)

    pl.plot(x1, final2[5])
    # pl.plot(x1, final2[1],'o')
    plt.title('y-position of a feature after butterworth')
    plt.xlabel('sample number')
    plt.ylabel('y-position')
    plt.grid(True)
    # pl.show()

    # print("final2 new", final2.shape)

    # -------------------------------------telos 3_1-----------------
    final_mean = np.array([])
    final_mean_value = np.array([])
    for i in range(0, final2.shape[1]):
        mean_value = np.array([])
        mean_value = np.mean(final21, axis=0)

    covariance_y = np.cov(final2)
    results = PCA()
    results.fit(covariance_y)
    eigenvalues = results.explained_variance_

    s0 = np.matmul(final2.T, results.components_[0])
    # print("final2rs y shape :",final2.shape)
    # print("final2rs y  T shape :",final2.T.shape)
    # print(" so shape :",s0.shape)
    # print("result:",results.components_[0].shape)

    s1 = np.matmul(final2.T, results.components_[1])
    # print("final2rs y shape :",final2.shape)
    # print("final2rs y  T shape :",final2.T.shape)
    # print(" s1 shape :",s1.shape)
    # print("result:",results.components_[1].shape)

    s2 = np.matmul(final2.T, results.components_[2])
    # print("final2rs y shape :",final2.shape)
    # print("final2rs y  T shape :",final2.T.shape)
    # print(" s1 shape :",s2.shape)
    # print("result:",results.components_[2].shape)

    s3 = np.matmul(final2.T, results.components_[3])
    # print("final2rs y shape :",final2.shape)
    # print("final2rs y  T shape :",final2.T.shape)
    # print(" s1 shape :",s3.shape)
    # print("result:",results.components_[3].shape)

    s4 = np.matmul(final2.T, results.components_[4])
    # print("final2rs y shape :",final2.shape)
    # print("final2rs y  T shape :",final2.T.shape)
    # print(" s1 shape :",s4.shape)
    # print("result:",results.components_[4].shape)

    x1 = range(s1.shape[0])
    # print(x1)

    pl.show()
    pl.plot(x1, s0)
    plt.title('$s_1(t)=M\cdot \phi_1$')
    plt.xlabel('sample number')
    plt.ylabel('y-position')
    plt.grid(True)
    pl.show()

    pl.plot(x1, s1)
    plt.title('$s_2(t)=M\cdot \phi_2$')
    plt.xlabel('sample number')
    plt.ylabel('y-position')
    plt.grid(True)
    pl.show()

    pl.plot(x1, s2)
    plt.title('$s_3(t)=M\cdot \phi_3$')
    plt.xlabel('sample number')
    plt.ylabel('y-position')
    plt.grid(True)
    pl.show()

    pl.plot(x1, s3)
    plt.title('$s_4(t)=M\cdot \phi_4$')
    plt.xlabel('sample number')
    plt.ylabel('y-position')
    plt.grid(True)
    pl.show()

    pl.plot(x1, s4)
    plt.title('$s_5(t)=M\cdot \phi_5$')
    plt.xlabel('sample number')
    plt.ylabel('y-position')
    plt.grid(True)
    pl.show()

    # ------------------------------------arxh 3_2------------------"""

    y = np.array([])
    butter_array = np.array([])
    fs = ((1 + interpol) * frames_1) / (end - start)
    print("fs=", fs)
    lowcut = 0.8
    highcut = 2.8
    final_butter2 = np.array([])
    # Number of samplepoints
    N = (1 + interpol) * frames_1
    # sample spacing
    T = (end - start) / N
    print("T=", T)
    # ------------------------------------------------------------------------
    # print(" so shape :",s0.shape)
    # print(" so before butterworth :",s0)
    y = s0
    # print(" so after butterworth :",y)
    # ------------------
    # print(y.shape)
    # x1=range(y.shape[0])
    # print(x1)
    # pl.plot(x1, y)
    # pl.plot(x1, final2[1],'o')
    # plt.title('y-position of a feature after butterworth')
    # plt.xlabel('sample number')
    # plt.ylabel('y-position')
    # plt.grid(True)
    # pl.show()

    '''
    Y = np.fft.fft(s0)
    freq = np.fft.fftfreq(len(s0), (len(s0)/(end-start)))

    plt.plot( freq, np.abs(Y) )
    plt.show()
    print("skata=",freq[np.argmax(np.abs(Y))])
    '''
    plt.show()
    # ------------------
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    # print("xf",xf.shape)
    # print("yf",yf.shape)

    fig, ax = plt.subplots()
    y2f = 2.0 / N * np.abs(yf[:N // 2])
    ax.plot(xf, y2f)

    plt.title('FFT of $s_1$')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('|Y|')
    plt.grid(True)
    plt.xlim([0, 7])
    max_position0 = np.argmax(y2f)
    print("y2f position0 einai =", max_position0)
    # print("y2f einai =", y2f)
    max_value0 = max(y2f)
    print("max to yf =", max_value0)
    freq_xf0 = xf[max_position0]
    print("h syxnothta xf0 =", freq_xf0)

    palmos0 = (60 / freq_xf0)
    print("o palmos0 einai-----------------------------------------------------------", palmos0)
    plt.show()

    # ----------------------------------------------------------------------------

    y = s1
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    print("xf", xf.shape)
    print("yf", yf.shape)
    fig, ax = plt.subplots()

    y2f = 2.0 / N * np.abs(yf[:N // 2])
    ax.plot(xf, y2f)

    plt.title('FFT of $s_2$')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('|Y|')
    plt.grid(True)
    plt.xlim([0, 7])
    max_position1 = np.argmax(y2f)
    print("y2f position1 einai =", max_position1)
    # print("y2f einai =", y2f)
    max_value1 = max(y2f)
    print("max to yf =", max_value1)
    freq_xf1 = xf[max_position1]
    print("h syxnothta xf =", freq_xf1)
    palmos1 = ((end-start) / freq_xf1) * (60/ (end-start))
    print("o palmos1 einai-----------------------------------------------------------", palmos1)
    plt.show()

    # -----------------------------------------------------------------------
    y = s2
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    print("xf", xf.shape)
    print("yf", yf.shape)
    fig, ax = plt.subplots()
    y2f = 2.0 / N * np.abs(yf[:N // 2])
    ax.plot(xf, y2f)

    plt.title('FFT of $s_3$')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('|Y|')
    plt.grid(True)
    plt.xlim([0, 7])
    max_position2 = np.argmax(y2f)
    print("y2f position einai2 =", max_position2)
    # print("y2f einai 2=", y2f)
    max_value2 = max(y2f)
    print("max to yf 2=", max_value2)
    freq_xf2 = xf[max_position2]
    print("h syxnothta 2xf =", freq_xf2)
    palmos2 = (60 / freq_xf2)
    print("o palmos2 einai-----------------------------------------------------------", palmos2)

    plt.show()

    # ------------------------------------------------------------------------
    y = s3
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    print("xf", xf.shape)
    print("yf", yf.shape)
    fig, ax = plt.subplots()
    y2f = 2.0 / N * np.abs(yf[:N // 2])
    ax.plot(xf, y2f)

    plt.title('FFT of $s_4$')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('|Y|')
    plt.grid(True)
    plt.xlim([0, 7])
    max_position3 = np.argmax(y2f)
    print("y2f position einai 3=", max_position3)
    # print("y2f einai3 =", y2f)
    max_value3 = max(y2f)
    print("max to yf 3=", max_value3)
    freq_xf3 = xf[max_position3]
    print("h syxnothta xf 3=", freq_xf3)
    palmos3 = (60 / freq_xf3)
    print("o palmos einai3-----------------------------------------------------------", palmos3)
    print("")
    #plt.show()

    # ----------------------------------------------------------------------------------
    plt.show()
    y = s4
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    print("xf", xf.shape)
    print("yf", yf.shape)
    fig, ax = plt.subplots()
    y2f = 2.0 / N * np.abs(yf[:N // 2])
    ax.plot(xf, y2f)

    plt.title('FFT of $s_5$')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('|Y|')
    plt.grid(True)
    plt.xlim([0, 7])

    max_position4 = np.argmax(y2f)
    print("y2f position einai4 =", max_position4)
    # print("y2f einai4 =", y2f)
    max_value4 = max(y2f)
    print("max to yf4 =", max_value4)
    freq_xf4 = xf[max_position4]
    print("h syxnothta xf4 =", freq_xf4)
    palmos4 = (60 / freq_xf4)
    print("o palmos einai4-----------------------------------------------------------", palmos4)
    print("")
    plt.show()

    #palmos= (palmos0 + palmos1 + palmos2 + palmos3 + palmos4) / 5
    #print(palmos)

    return palmos0

def cal_cf_chest(cfx_face, cfy_face, cfx_teliko_array, cfy_teliko_array, frames_1):

    round_array1 = np.array([])
    kinhsh_array1 = np.array([])
    max_distance_array1 = np.array([])
    cfx_teliko = np.array([])
    for i in range(0, cfx_face.shape[0]):

        x_axis1 = np.arange(0, frames_1, 1)
        y_axis1 = np.array([])
        interpolation_x = np.array([])

        max_distance1 = np.array([])
        kinhsh_temp_array1 = np.array([])

        for j in range(0, counter_while - 1):
            # sto y_axis kanw append- bazw thn kathe grammh
            y_axis1 = np.append(y_axis1, cfx_face[i, j])
        # to y_axis exei gemisei me thn prwth grammh kanw reshape
        # kanw cubic interpolation se kathe grammh kai reshape sta new x kai y
        tck1 = interpolate.splrep(x_axis1, y_axis1, s=0)
        x_axis_new1 = np.arange(0, frames_1, 0.125)
        y_axis_new1 = interpolate.splev(x_axis_new1, tck1, der=0)

        if i == 0:
            interpolation_array1 = np.zeros(y_axis_new1.shape)
            interpolation_array1 = interpolation_array1.reshape((-1, 1))

            kinhsh_array1 = np.zeros(counter_while - 1)
            kinhsh_array1 = kinhsh_array1.reshape((-1, 1))
        # sto interpolation_y thelw na kanw append tis times twn y gia kathe grammi ta opoia exoun ginei perisotera logo ts interpolation
        interpolation_x = np.append(interpolation_x, y_axis_new1)
        interpolation_x = np.matrix(interpolation_x)
        interpolation_x = interpolation_x.T
        # ston interpolation pinaka thelw na ftiaksw ton teliko pinaka NxN me ta stoixeia logo ths interpolation
        interpolation_array1 = np.concatenate((interpolation_array1, interpolation_x), axis=1)
        # print("inteer",interpolation_array.shape)
        # print(interpolation_y)
        for j in range(0, counter_while - 1):
            kinhsh1 = np.fabs(cfx_face[i, j] - cfx_face[i, j + 1])

            kinhsh_temp_array1 = np.append(kinhsh_temp_array1, kinhsh1)
        # exeis shape (1,)...(5,)

        # to max distance tha exei to megisto kai o array tha ta exei ola
        max_distance1 = np.amax(kinhsh_temp_array1)
        max_distance_array1 = np.append(max_distance_array1, max_distance1)
        # o round tha ta exei stoggulopoihmena
        round_array1 = np.around(max_distance_array1, decimals=1)

        # o pinakas kinhsh array tha exei to synoliko pinaka metakinhsewn
        kinhsh_temp_array1 = kinhsh_temp_array1.reshape((-1, 1))
        # print("tem shape",kinhsh_temp_array)
        kinhsh_array1 = np.concatenate((kinhsh_array1, kinhsh_temp_array1), axis=1)

    interpolation_array1 = interpolation_array1.T

    timh_katanomhs1 = stats.mode(round_array1)
    # print("h epanemfanizomenh timh einai  toy x h:",timh_katanomhs1[0])

    # h where returns ndarray or tuple of ndarrays
    cfx_teliko = np.where(round_array1 <= timh_katanomhs1[0])

    # to cfy_teliko exei tiw theseis twn stoixeiwn toy y gia to cfx poy xreiazomatse
    cfx_teliko_array = np.append(cfx_teliko_array, cfx_teliko[0])

    interpolation_array1 = interpolation_array1.astype(int)
    cfx_teliko_array = cfx_teliko_array.astype(int)

    final21 = np.array([])

    for i in range(0, cfx_teliko_array.shape[0]):  # apo 0 ews ta telika y poy thelw

        final1 = np.array([])
        final1 = np.append(final1, interpolation_array1[cfx_teliko_array[i]])
        final21 = np.concatenate((final21, final1), axis=0)

    final_reshape = np.reshape(final21, (cfx_teliko_array.shape[0], -1))

    # ---------- mean kai covariance-----------ζ
    final_mean1 = np.array([])
    final_mean_value1 = np.array([])
    for i in range(0, final_reshape.shape[1]):
        mean_value1 = np.array([])

        mean_value1 = np.mean(final_reshape, axis=0)

    covariance_x = np.cov(final_reshape)

    # telos tou x

    # interpolation_array=np.array([])

    round_array = np.array([])
    kinhsh_array = np.array([])
    max_distance_array = np.array([])
    cfy_teliko = np.array([])

    for i in range(0, cfy_face.shape[0]):

        x_axis = np.arange(0, frames_1, 1)
        y_axis = np.array([])
        interpolation_y = np.array([])

        max_distance = np.array([])
        kinhsh_temp_array = np.array([])

        for j in range(0, counter_while - 1):
            # sto y_axis kanw append- bazw thn kathe grammh
            y_axis = np.append(y_axis, cfy_face[i, j])
        # to y_axis exei gemisei me thn prwth grammh kanw reshape
        # kanw cubic interpolation se kathe grammh kai reshape sta new x kai y
        tck = interpolate.splrep(x_axis, y_axis, s=0)
        x_axis_new = np.arange(0, frames_1, 0.125)
        y_axis_new = interpolate.splev(x_axis_new, tck, der=0)

        if i == 0:
            interpolation_array = np.zeros(y_axis_new.shape)
            interpolation_array = interpolation_array.reshape((-1, 1))

            kinhsh_array = np.zeros(counter_while - 1)
            kinhsh_array = kinhsh_array.reshape((-1, 1))
        # sto interpolation_y thelw na kanw append tis times twn y gia kathe grammi ta opoia exoun ginei perisotera logo ts interpolation
        interpolation_y = np.append(interpolation_y, y_axis_new)
        interpolation_y = np.matrix(interpolation_y)
        interpolation_y = interpolation_y.T
        # ston interpolation pinaka thelw na ftiaksw ton teliko pinaka NxN me ta stoixeia logo ths interpolation
        interpolation_array = np.concatenate((interpolation_array, interpolation_y), axis=1)

        for j in range(0, counter_while - 1):
            kinhsh = np.fabs(cfy_face[i, j] - cfy_face[i, j + 1])

            kinhsh_temp_array = np.append(kinhsh_temp_array, kinhsh)
        # exeis shape (1,)...(5,)

        # to max distance tha exei to megisto kai o array tha ta exei ola
        max_distance = np.amax(kinhsh_temp_array)
        max_distance_array = np.append(max_distance_array, max_distance)
        # o round tha ta exei stoggulopoihmena
        round_array = np.around(max_distance_array, decimals=1)

        # o pinakas kinhsh array tha exei to synoliko pinaka metakinhsewn
        kinhsh_temp_array = kinhsh_temp_array.reshape((-1, 1))
        kinhsh_array = np.concatenate((kinhsh_array, kinhsh_temp_array), axis=1)

    interpolation_array = interpolation_array.T

    timh_katanomhs = stats.mode(round_array)

    # h where returns ndarray or tuple of ndarrays
    cfy_teliko = np.where(round_array <= timh_katanomhs[0])

    # to cfy_teliko exei tiw theseis twn stoixeiwn toy y gia to cfx poy xreiazomatse
    cfy_teliko_array = np.append(cfy_teliko_array, cfy_teliko[0])


    interpolation_array = interpolation_array.astype(int)
    cfy_teliko_array = cfy_teliko_array.astype(int)

    final2 = np.array([])

    for i in range(0, cfy_teliko_array.shape[0]):  # apo 0 ews ta telika y poy thelw

        final = np.array([])
        final = np.append(final, interpolation_array[cfy_teliko_array[i]])
        final2 = np.concatenate((final2, final), axis=0)

    final_reshape2 = np.reshape(final2, (cfy_teliko_array.shape[0], -1))

    #print(final_reshape2.shape, interpolation_array.shape)

    final_mean = np.array([])
    final_mean_value = np.array([])
    for i in range(0, final_reshape2.shape[1]):
        mean_value = np.array([])

        mean_value = np.mean(final_reshape2, axis=0)

    covariance_y = np.cov(final_reshape2)

    results = PCA()
    results.fit(covariance_y)
    eigenvalues = results.explained_variance_
   #print(final_reshape2)
   #print(final_reshape.T)
    #print(results)

    s0 = np.matmul(final_reshape2.T, results.components_[0])
    print("\nfinal2rs y shape :", final_reshape2.shape)
    print("final2rs y  T shape :", final_reshape2.T.shape)
    print(" s1 shape :", s0.shape)
    print("result:", results.components_[0].shape)

    #print(len(s0))
    #print(s0.shape)

    #print(s0)

    s0_val_min= s0[0]
    s0_val_max = s0[0]
    counter_breaths= 0
    point= 1

    r = np.ptp(s0, axis=0)
    print("print r= ", r)

    if r < 10:
        T = 0.9
    elif r < 30:
        T = 1.5
    elif r < 60:
        T = 4
    else:
        T = 6

    for i in range(len(s0)):
        if s0[i]> s0_val_max:
            s0_val_max= s0[i]
            point=0
            #print(s0_val_max)
        if s0[i]< s0_val_min:
            s0_val_min= s0[i]
            s0_val_max= s0[i]
            #print("s0_val_min ",+s0_val_min)
        if s0[i]< s0_val_max-T and point== 0:
            counter_breaths+=1
            s0_val_min=s0[i]
            point=1



    #s1 = np.matmul(final_reshape2.T, results.components_[1])

    x1 = range(s0.shape[0])
    # print(x1)
    pl.plot(x1, s0)
    plt.title('chest')
    pl.show()

    y = np.array([])
    butter_array = np.array([])
    fs = 200
    lowcut = 0.1
    highcut = 0.4

    final_butter2 = np.array([])

    N = 600
    # sample spacing
    T = 1.0 / 800.0
    # y = final_butter2

    y1 = s0
    y = butter_bandpass_filter(y1, lowcut, highcut, fs, order=5)

    yf = scipy.fftpack.fft(y)
    xf = (np.linspace(0.0, 1.0 // (2.0 * T), N // 2))
    #print("xf", xf.shape)
    #print("yf", yf.shape)
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.title('s0_chest')
    plt.show()

    dft_s0 = cv2.dft(np.float32(s0), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_s0_1 = dft_s0[:, :, 0]
    dft_s0_abs = np.fabs(dft_s0_1)
    # print("dft=", dft_s0_abs)
    pow_s0 = np.power(dft_s0_abs, 2)

    max_value = np.argmax(pow_s0)
    #print("max=", max_value)
    #print("")

    print("\n")
    print("Counter_breaths= ",+counter_breaths)

    return counter_breaths

if __name__ == '__main__':
    counter_while= 0
    counter = 0
    counter_refresh = 0
    counter_hr= 0

    pygame.init()
    size= [640,480]
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    FPS =10
    done= False
    point= False
    screen_info=True

    is_working= True

    pal= 0
    breaths= 0

    font = pygame.font.SysFont("comicsansms", 32)
    heart = pygame.image.load('heart.png')
    lung = pygame.image.load('lung.png')

    start= time.time()
    faces,chest,first_frame,first_frame_gray,lk_params,corners_face_arr,corners_chest_arr,roi_first_gray_chest,roi_first_color_chest,roi_first_gray_face,roi_first_color_face,mask_face,mask_chest = initial()

    while not done:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    done = True
                if event.key == pygame.K_1:
                    if point== True:
                        point = False
                    else:
                        point= True
                if event.key == pygame.K_2:
                    if screen_info== True:
                        screen_info = False
                    else:
                        screen_info= True
                if event.key == pygame.K_c and is_working == False:

                    cfx_face = np.array([])
                    cfy_face = np.array([])
                    cfx_chest = np.array([])
                    cfy_chest = np.array([])

                    cfx_teliko_array = np.array([])
                    cfy_teliko_array = np.array([])

                    cfx_teliko_array_chest = np.array([])
                    cfy_teliko_array_chest = np.array([])

                    counter_while = 0
                    counter = 0
                    counter_refresh = 0
                    counter_hr = 0
                    pal = 0
                    breaths = 0

                    start = time.time()

                    is_working = True

                    cap = cv2.VideoCapture(0)

                    faces, chest, first_frame, first_frame_gray, lk_params, corners_face_arr, corners_chest_arr, roi_first_gray_chest, roi_first_color_chest, roi_first_gray_face, roi_first_color_face, mask_face, mask_chest = initial()

        #print(counter)
        #print(counter_refresh)

        if is_working== True:

            if counter_refresh == 600:

                counter_while= 0

                cfx_face = np.array([])
                cfy_face = np.array([])
                cfx_chest = np.array([])
                cfy_chest = np.array([])

                cfx_teliko_array = np.array([])
                cfy_teliko_array = np.array([])

                cfx_teliko_array_chest = np.array([])
                cfy_teliko_array_chest = np.array([])

                counter_refresh= 0
                pal = 0
                breaths = 0

                start=time.time()

            try:
                corners_face_arr,corners_chest_arr,cfx_face,cfy_face,cfx_chest,cfy_chest,frame, roi_first_gray_face, roi_first_gray_chest, frame = tracking(roi_first_gray_face, corners_face_arr, lk_params, roi_first_gray_chest, corners_chest_arr, mask_face,mask_chest, counter_while, cfx_face,cfy_face,cfx_chest,cfy_chest, point)
            except Exception as e:
                #faces,chest,first_frame,first_frame_gray,lk_params,corners_face_arr,corners_chest_arr,roi_first_gray_chest,roi_first_color_chest,roi_first_gray_face,roi_first_color_face,mask_face,mask_chest = initial()
                counter_hr= 0
                counter= 0
                counter_refresh= 0
                is_working= False
                print(e)
                continue

            counter_while+= 1
            counter+=1
            counter_refresh+= 1
            counter_hr+= 1

            #print(counter_refresh)

            if counter == 100:
                counter_breaths = cal_cf_chest(cfx_chest, cfy_chest, cfx_teliko_array_chest, cfy_teliko_array_chest, counter_while -1)
                counter= 0
                breaths = counter_breaths
            if counter_hr == 300:
                palmos0 = cal_cf_face(cfx_face, cfy_face, cfx_teliko_array, cfy_teliko_array, counter_while - 1, start)
                counter_hr= 0
                pal = int(palmos0)

            imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(imageRGB)

            mode = img.mode
            size = img.size
            data = img.tobytes()

            py_image= pygame.image.fromstring(data, size, mode)

            screen.blit(py_image,(0,0))

            if screen_info== True and pal==0 and breaths== 0:
                txt1=font.render(str(pal), True, WHITE)
                screen.blit(heart, [0,0])
                screen.blit(txt1, [70, 10])
                txt1 = font.render(str(breaths), True, WHITE)
                screen.blit(lung, [0, 50])
                screen.blit(txt1,[70,60])
            if screen_info== True and pal==0 and breaths!= 0:
                txt1=font.render(str(pal), True, WHITE)
                screen.blit(heart, [0,0])
                screen.blit(txt1, [70, 10])
                txt1 = font.render(str(breaths), True, WHITE)
                screen.blit(lung, [0, 50])
                screen.blit(txt1,[70,60])
            if screen_info== True and pal!=0 and breaths!= 0:
                txt1=font.render(str(pal), True, WHITE)
                screen.blit(heart, [0,0])
                screen.blit(txt1, [70, 10])
                txt1 = font.render(str(breaths), True, WHITE)
                screen.blit(lung, [0, 50])
                screen.blit(txt1,[70,60])

        else:
            screen.fill([255,0,0])

            cap= 0

            text = font.render(" Press the button c to continue ", True, WHITE)
            screen.blit(text, [100, 220])
            text = font.render(" and try to stay calm :) ", True, WHITE)
            screen.blit(text, [160, 250])

        clock.tick(FPS)
        pygame.display.flip()

    pygame.quit()

    #cal_cf_face(cfx_face, cfy_face, cfx_teliko_array, cfy_teliko_array, counter_while - 1,start)
    #cal_cf_chest(cfx_chest, cfy_chest, cfx_teliko_array_chest, cfy_teliko_array_chest, counter_while - 1)