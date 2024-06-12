import os
import cv2
import numpy as np
from argparse import ArgumentParser
import os
import live_inferencer
import sys

parser = ArgumentParser()
parser.add_argument("output", type=str, required=True)

args = parser.parse_args()

def create_intermediate_vid_name():
    current_analyzed_files = len(os.listdir(directory_path))
    return f"input_vid{current_analyzed_files+1}"

input_vidname = create_intermediate_vid_name()

if os.path.exists(os.path.join('analyzed_videos', args.output)):
    print("The output vidname you specified already exists!!")
    print("Please rerun this script with a unique output video name.")
    print("\n\nExiting...")
    sys.exit()

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(input_vidname, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Livestream Video Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

trt_model = live_inferencer.load_tenssort_engine("16workspace_size_64ms_inference.trt")
live_inferencer.create_output_vid(os.path.join('analyzed_videos', input_vidname), 
                  os.path.join('analyzed_videos', args.output))

live_inferencer.remove(input_vidname)




