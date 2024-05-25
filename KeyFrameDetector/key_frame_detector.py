import os
import cv2
import csv
import numpy as np
import time
import peakutils
from tqdm import tqdm
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics

def create_image_grid(images, keyframe_data, num_cols=3):
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    image_height, image_width, _ = images[0].shape
    
    grid_image = np.zeros((num_rows * image_height, num_cols * image_width, 3), dtype=np.uint8)
    
    max_diff_magnitude = max(keyframe_data, key=lambda x: x[3])[3]
    
    for i in range(num_images):
        row = i // num_cols
        col = i % num_cols
        
        image = images[i]
        keyframe_number, frame_number, timestamp, diff_magnitude = keyframe_data[i]
        diff_percentage = (diff_magnitude / max_diff_magnitude) * 100
        
        # Add keyframe information and difference magnitude as text
        text = f"Keyframe: {keyframe_number} | Frame: {frame_number} | Time: {timestamp:.2f}s | Diff: {diff_magnitude} ({diff_percentage:.2f}%)"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        grid_image[row*image_height:(row+1)*image_height, col*image_width:(col+1)*image_width] = image
    
    return grid_image

def create_grayframe_grid(grayframes, keyframe_data, num_cols=3):
    num_images = len(grayframes)
    num_rows = (num_images + num_cols - 1) // num_cols
    image_height, image_width = grayframes[0].shape
    
    grid_image = np.zeros((num_rows * image_height, num_cols * image_width), dtype=np.uint8)
    
    max_diff_magnitude = max(keyframe_data, key=lambda x: x[3])[3]
    
    for i in range(num_images):
        row = i // num_cols
        col = i % num_cols
        
        grayframe = grayframes[i]
        keyframe_number, frame_number, timestamp, diff_magnitude = keyframe_data[i]
        diff_percentage = (diff_magnitude / max_diff_magnitude) * 100
        
        # Add keyframe information and difference magnitude as text
        text = f"Keyframe: {keyframe_number} | Frame: {frame_number} | Time: {timestamp:.2f}s | Diff: {diff_magnitude} ({diff_percentage:.2f}%)"
        cv2.putText(grayframe, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        grid_image[row*image_height:(row+1)*image_height, col*image_width:(col+1)*image_width] = grayframe
    
    return grid_image

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
    keyframe_images = []
    keyframe_grayframes = []
    keyframe_data = []
    for x in indices:
        cv2.imwrite(os.path.join(keyframePath, 'keyframe' + str(cnt) + '.jpg'), full_color[x])
        keyframe_images.append(full_color[x])
        keyframe_grayframes.append(images[x])
        log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
        if verbose:
            print(log_message)

        # Write the keyframe data to the CSV file
        keyframe_info = (cnt, lstfrm[x], timeSpans[x], lstdiffMag[x])
        keyframe_data.append(keyframe_info)
        with open(path2file, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(keyframe_info)
        cnt += 1

    # Create image grid for color keyframes
    num_cols = 3  # Specify the desired number of columns (3 or 4)
    image_grid = create_image_grid(keyframe_images, keyframe_data, num_cols)
    cv2.imwrite(os.path.join(imageGridsPath, 'image_grid.jpg'), image_grid)

    # Create image grid for grayframe keyframes
    grayframe_grid = create_grayframe_grid(keyframe_grayframes, keyframe_data, num_cols)
    cv2.imwrite(os.path.join(imageGridsPath, 'grayframe_grid.jpg'), grayframe_grid)

    cv2.destroyAllWindows()

