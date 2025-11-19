import numpy as np
import cv2

# Hold the background frame for background subtraction.
background = None
# Hold the hand's data so all its details are in one place.
hand = None
# Variables to count how many frames have passed and to set the size of the window.
frames_elapsed = 0
FRAME_HEIGHT = 1000
FRAME_WIDTH = 1500
# Humans come in a ton of beautiful shades and colors.
# Try editing these if your program has trouble recognizing your skin tone.
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

capture = cv2.VideoCapture(0)

# Our region of interest will be the top right part of the frame.
region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH / 2)
region_right = FRAME_WIDTH

frames_elapsed = 0


class HandData:
    top = (0,0)
    bottom = (0,0)
    left = (0,0)
    right = (0,0)
    centerX = 0
    prevCenterX = 0
    isInFrame = False
    isWaving = False
    fingers = 0
    
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0

        #isWaving = False
        #isInFrame = False
        self.isInFrame = False
        self.isWaving = False

    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def check_for_waving(self, centerX):
        self.prevCenterX = self.centerX
        self.centerX = centerX
        
        if abs(self.centerX - self.prevCenterX) > 3:
            self.isWaving = True
        else:
            self.isWaving = False


# Here we take the current frame, the number of frames elapsed, and how many fingers we've detected
# so we can print on the screen which gesture is happening (or if the camera is calibrating).
def write_on_image(frame, hand):
    text = "Searching..."

    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."
    elif hand == None or hand.isInFrame == False:
        text = "No hand detected"
    else:
        if hand.isWaving:
            text = "Waving"
        elif hand.fingers == 0:
            text = "Rock"
        elif hand.fingers == 1:
            text = "Pointing"
        elif hand.fingers == 2:
            text = "Scissors"
    
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,( 0 , 0 , 0 ),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)

    # Highlight the region of interest using a drawn rectangle.
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255,255,255), 2)


def get_region(frame):
    # Separate the region of interest from the rest of the frame.
    region = frame[region_top:region_bottom, region_left:region_right]
    # Make it grayscale so we can detect the edges more easily.
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Use a Gaussian blur to prevent frame noise from being labeled as an edge.
    region = cv2.GaussianBlur(region, (5,5), 0)

    return region

def get_average(region):
    # We have to use the global keyword because we want to edit the global variable.
    global background
    # If we haven't captured the background yet, make the current region the background.
    if background is None:
        background = region.copy().astype("float")
        return
    # Otherwise, add this captured frame to the average of the backgrounds.
    cv2.accumulateWeighted(region, background, BG_WEIGHT)





# Here we use differencing to separate the background from the object of interest.
def segment(region):
    global hand
    # Find the absolute difference between the background and the current frame.
    diff = cv2.absdiff(background.astype(np.uint8), region)

    # Threshold that region with a strict 0 or 1 ruling so only the foreground remains.
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # Get the contours of the region, which will return an outline of the hand.
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If we didn't get anything, there's no hand.
    if len(contours) == 0:
        if hand is not None:
            hand.isInFrame = False
        return
    # Otherwise return a tuple of the filled hand (thresholded_region), along with the outline (segmented_region).
    else:
        if hand is not None:
            hand.isInFrame = True
        segmented_region = max(contours, key = cv2.contourArea)
        return (thresholded_region, segmented_region)
    
    
    


def get_hand_data(thresholded_image, segmented_image):
    global hand

    # If segmented_image is a binary mask, extract the largest contour first
    # to ensure we get a contour suitable for convexHull().
    if len(segmented_image.shape) == 2:
        contours, _ = cv2.findContours(segmented_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return  # Nothing found
        segmented_image = max(contours, key=cv2.contourArea)

    # Compute the convex hull around the segmented hand
    convexHull = cv2.convexHull(segmented_image)

    # Find extremal points
    top    = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left   = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right  = tuple(convexHull[convexHull[:, :, 0].argmax()][0])

    # Compute the palm's horizontal center (used for waving detection)
    centerX = int((left[0] + right[0]) / 2)

    # Initialize or update the global HandData object
    if hand is None:
        hand = HandData(top, bottom, left, right, centerX)
    else:
        hand.update(top, bottom, left, right)

    # Every few frames, check for waving
    if frames_elapsed % 8 == 0:
        hand.check_for_waving(centerX)

    # Draw convex hull and key points for visualization
    hull_visual = np.zeros((thresholded_image.shape[0], thresholded_image.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(hull_visual, [convexHull], -1, (0, 255, 0), 2)  # green hull

    # Draw extremities as colored dots
    cv2.circle(hull_visual, top, 5, (0, 0, 255), -1)     # top - red
    cv2.circle(hull_visual, bottom, 5, (255, 0, 0), -1)  # bottom - blue
    cv2.circle(hull_visual, left, 5, (0, 255, 255), -1)  # left - yellow
    cv2.circle(hull_visual, right, 5, (255, 255, 0), -1) # right - cyan

    # Display the convex hull window
    cv2.imshow("Convex Hull", hull_visual)




while (True):
    # Store the frame from the video capture and resize it to the desired window size.
    ret, frame = capture.read()
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.
    frame = cv2.flip(frame, 1)

    # Separate the region of interest and prep it for edge detection.
    region = get_region(frame)

    if frames_elapsed < CALIBRATION_TIME:
        get_average(region)
    else:
        region_pair = segment(region)
        if region_pair is not None:
            # If we have the regions segmented successfully, show them in another window for the user.
            (thresholded_region, segmented_region) = region_pair
            cv2.drawContours(region, [segmented_region], -1, (255, 255, 255))
            cv2.imshow("Segmented Image", region)
            
            get_hand_data(thresholded_region, segmented_region)

    # Write the action the hand is doing on the screen, and draw the region of interest.
    write_on_image(frame, hand)

    cv2.imshow("Camera Input", frame)
   
    frames_elapsed += 1

     # Check if user wants to exit.
    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break

# When we exit the loop, we have to stop the capture too.
capture.release()
cv2.destroyAllWindows()