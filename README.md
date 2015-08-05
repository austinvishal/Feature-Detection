# Feature-Detection
The goal of the lab is to track a white bulb in a video using openCV’s feature detection and matching tools
The tracking is shown by displaying a rectangle which encloses the bulb along the video. This rectangle shows the orientation angle of the bulb with respect to the original frame. The following methods are implemented using the SURF detector:
1. Matching the object in current frame to the object in the first frame of the video
2. Matching the object in current frame to the object in the previous frame of the video
