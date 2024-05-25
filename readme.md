<center>

   ![header](images/header.png)
    
</center>

A `Key Frame` is a location on a video timeline which marks the beginning or end of a smooth transition throughout the fotograms, `Key Frame Detector` try to look for the most representative and significant frames that can describe the movement or main events in a video using peakutils peak detection functions.

<br/>
<p align="center">

   <img src="images/demo.gif"> 

</p>
<br/>

## Installation

**Requirements**

- python3
- numpy
- opencv
- peakutils
- matplotlib
- PIL
  
<hr />
```
# Video Keyframe Detector

A Key Frame is a location on a video timeline which marks the beginning or end of a smooth transition throughout the frames. Key Frame Detector tries to look for the most representative and significant frames that can describe the movement or main events in a video using peakutils peak detection functions.

## Installation

### Requirements

- python3
- numpy
- opencv
- peakutils
- matplotlib
- PIL
- tqdm

### Installing Dependencies

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

To use the Key Frame Detector, you can run the `cli.py` script with the following command-line arguments:

```
python cli.py --source="path/to/your/video.mp4" --dest="output/directory" --Thres=0.3 --verbose
```

- `--source`: Path to the input video file.
- `--dest`: Directory where the output keyframes and CSV file will be saved.
- `--Thres`: Threshold value for peak detection (default: 0.3).
- `--verbose`: Optional flag to enable verbose output.

Example:

```
python cli.py --source="videos/acrobacia.mp4" --dest="out" --Thres=0.3 --verbose
```

This command will process the video file "acrobacia.mp4" located in the "videos" directory, save the detected keyframes and CSV file in the "out" directory, use a threshold value of 0.3 for peak detection, and provide verbose output.

## Output

The Key Frame Detector will generate the following output:

- Keyframes: The detected keyframes will be saved as individual image files in the specified output directory.
- CSV File: A CSV file named "output.csv" will be created in the output directory, containing information about each detected keyframe, including the keyframe number, frame number, timestamp (in seconds), and difference magnitude.

## Progress Bar

During the keyframe detection process, a progress bar will be displayed to track the progress. The progress bar shows the current frame being processed, the total number of frames, and the processing speed in frames per second.

## Additional Features

- `--plotMetrics`: Optional flag to plot metrics for monitoring the selected keyframes.
```






