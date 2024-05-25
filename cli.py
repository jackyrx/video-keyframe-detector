import argparse

from KeyFrameDetector.key_frame_detector import keyframeDetection

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--source', help='source file', required=True)
    parser.add_argument('-d', '--dest', help='destination folder', required=True)
    parser.add_argument('-t','--Thres', help='Threshold of the image difference', default=0.3)
    parser.add_argument('-v', '--verbose', help='print verbose', action='store_true')

    args = parser.parse_args()


    keyframeDetection(args.source, args.dest, float(args.Thres), args.verbose)

if __name__ == '__main__':
    main()
