import os
import cv2
import csv
import numpy as np
import time
import peakutils
from tqdm import tqdm
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics

def keyframeDetection(source, dest, Thres, plotMetrics=False, verbose=False):
    keyframePath = dest + '/keyFrames'
    imageGridsPath = dest + '/imageGrids'
    csvPath = dest + '/csvFile'
    path2file = csvPath + '/output.csv'
    prepare_dirs(keyframePath, imageGridsPath, csvPath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video file")
        return

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None
    Start_time = time.process_time()

    # Create a progress bar
    progress_bar = tqdm(total=length, unit='frames', desc='Processing')

    # Read until video is completed
    for i in range(length):
        ret, frame = cap.read()
        grayframe, blur_gray = convert_frame_to_grayscale(frame)

        if blur_gray is None:
            break
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time - Start_time
        timeSpans.append(time_Span)
        lastFrame = blur_gray

        # Update the progress bar
        progress_bar.update(1)

    cap.release()
    progress_bar.close()

    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y - base, Thres, min_dist=1)

    # Plot to monitor the selected keyframe
    if plotMetrics:
        print('Plotting metrics...')
        plot_metrics(indices, lstfrm, lstdiffMag, y)

    # Initialize the CSV file with column names
    csv_columns = ['Keyframe', 'Frame Number', 'Timestamp (sec)', 'Difference Magnitude']
    with open(path2file, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(csv_columns)

    cnt = 1
    for x in indices:
        cv2.imwrite(os.path.join(keyframePath, 'keyframe' + str(cnt) + '.jpg'), full_color[x])
        log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
        if verbose:
            print(log_message)

        # Write the keyframe data to the CSV file
        keyframe_data = [cnt, lstfrm[x], timeSpans[x], lstdiffMag[x]]
        with open(path2file, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(keyframe_data)
        cnt += 1

    cv2.destroyAllWindows()