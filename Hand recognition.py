import numpy as np
import cv2


frames_elapsed = 0

FRAME_HEIGHT = 600
FRAME_WIDTH = 800

CALIBRATION_TIME = 40
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

#global used by loop
capture = cv2.VideoCapture(0)


'''
We need two separate regions detected to account for both the right and the left hand
Workflow:
- initialize the two hand data and store the results in the global Hand variable
- define two regions of the screen which will be key to 
- define the class HandData which works for both hands


'''


ROI_HEIGHT = 250 
ROI_WIDTH  = 200 

# Left ROI
L_region_top = 40
L_region_bottom = L_region_top + ROI_HEIGHT
L_region_left = 30
L_region_right = L_region_left + ROI_WIDTH

# Right ROI
R_region_top = 40
R_region_bottom = R_region_top + ROI_HEIGHT
R_region_left = FRAME_WIDTH - ROI_WIDTH - 30
R_region_right = R_region_left + ROI_WIDTH



frames_elapsed = 0

# background images and accumulators (initialized at runtime after we know ROI shapes)
background_left  = None
background_right = None
bg_acc_left      = None
bg_acc_right     = None

hand_left  = None
hand_right = None


class HandData:
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = centerX
        self.isInFrame = True
        self.isWaving = False
        self.fingers = 0

    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def check_for_waving(self, centerX, threshold=3):
        self.prevCenterX = self.centerX
        self.centerX = centerX
        self.isWaving = abs(self.centerX - self.prevCenterX) > threshold

def get_region(frame, top, bottom, left, right):
    """Return a blurred grayscale ROI from the frame"""
    region = frame[top:bottom, left:right]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, region  # return both the gray mask and the colored crop (for visualization)


def get_average_accumulator(gray_region, bg_acc):
    """Add gray_region to accumulator and initialize if needed"""
    if bg_acc is None:
        bg_acc = np.zeros_like(gray_region, dtype=np.float32)
    bg_acc += gray_region.astype(np.float32)
    return bg_acc


def finalize_background_from_acc(bg_acc):
    """Compute final background average from accumulator"""
    if bg_acc is None:
        return None
    return (bg_acc / CALIBRATION_TIME).astype(np.uint8)


def safe_find_contours(binary):
    """Return contours robustly across OpenCV versions"""
    results = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(results) == 2:
        contours = results[0]
    else:
        contours = results[1]
    return contours
  

def segment(region, background):
    """
    Returns (thresholded_region, largest_contour)
    OR None if no segmentation is possible.
    """

    # If background is not initialized, we cannot segment.
    if background is None:
        return None

    # Compute absolute difference between background and current region.
    diff = cv2.absdiff(background.astype(np.uint8), region)

    # Threshold foreground
    _, thresholded_region = cv2.threshold(
        diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresholded_region.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Largest contour = hand
    largest = max(contours, key=cv2.contourArea)

    return thresholded_region, largest

    
    


# def get_hand_data(thresholded_image, segmented_contour, hand_obj, debug_prefix=""):
#     """build convex hull, find extremal points, show hull debug window.
#        Returns updated HandData object (or the same if input is None)."""
#     # ensure we have a contour
#     if segmented_contour is None:
#         return None

#     # convex hull of the contour (returns points)
#     convexHull = cv2.convexHull(segmented_contour)

#     # Extremal points
#     top    = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
#     bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
#     left   = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
#     right  = tuple(convexHull[convexHull[:, :, 0].argmax()][0])

#     # center X (palm horizontal center)
#     centerX = int((left[0] + right[0]) / 2)

#     # update or create HandData
#     if hand_obj is None:
#         hand_obj = HandData(top, bottom, left, right, centerX)
#     else:
#         hand_obj.update(top, bottom, left, right)
#         # check waving occasionally (we'll call every frame or from main loop)
#         hand_obj.check_for_waving(centerX)

#     # Build visualization for hull: use color image of same size as thresholded
#     h, w = thresholded_image.shape[:2]
#     hull_visual = np.zeros((h, w, 3), dtype=np.uint8)
#     cv2.drawContours(hull_visual, [convexHull], -1, (0, 255, 0), 2)

#     # draw extremal points
#     cv2.circle(hull_visual, top, 5, (0, 0, 255), -1)
#     cv2.circle(hull_visual, bottom, 5, (255, 0, 0), -1)
#     cv2.circle(hull_visual, left, 5, (0, 255, 255), -1)
#     cv2.circle(hull_visual, right, 5, (255, 255, 0), -1)
#     # draw center line
#     cv2.line(hull_visual, (centerX, 0), (centerX, h), (200, 200, 200), 1)

#     # Show debug window with distinct name if desired
#     winname = f"{debug_prefix}Convex Hull"
#     cv2.imshow(winname, hull_visual)

#     return hand_obj

def get_hand_data(thresholded_image, segmented_contour, hand_obj, debug_prefix=""):
    if segmented_contour is None:
        return None

    # Convex hull (points)
    convexHull = cv2.convexHull(segmented_contour)

    hull_indices = cv2.convexHull(segmented_contour, returnPoints=False)

    # -----------------------
    # SAFETY BLOCK TO FIX ERROR
    # -----------------------
    defects = None
    # convexityDefects requires:
    # - at least 3 hull indices
    # - at least 3 contour points
    # - a non-self-intersecting contour
    if hull_indices is not None \
    and len(hull_indices) > 3 \
    and segmented_contour is not None \
    and len(segmented_contour) > 3:

        try:
            defects = cv2.convexityDefects(segmented_contour, hull_indices)
        except cv2.error:
            # if convexityDefects fails, silently ignore
            defects = None


    # Extremal points
    top    = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left   = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right  = tuple(convexHull[convexHull[:, :, 0].argmax()][0])

    centerX = int((left[0] + right[0]) / 2)

    # update/create handObj
    if hand_obj is None:
        hand_obj = HandData(top, bottom, left, right, centerX)
    else:
        hand_obj.update(top, bottom, left, right)
        hand_obj.check_for_waving(centerX)

    # -------------------------
    # FINGER COUNTING SECTION
    # -------------------------
    finger_count = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, depth = defects[i, 0]
            start = tuple(segmented_contour[s][0])
            end   = tuple(segmented_contour[e][0])
            far   = tuple(segmented_contour[f][0])

            # Reject small defects (noise)
            if depth < 1500:   # tune if needed
                continue

            # Compute angle between fingers
            a = np.linalg.norm(np.array(end) - np.array(far))
            b = np.linalg.norm(np.array(start) - np.array(far))
            c = np.linalg.norm(np.array(start) - np.array(end))
            angle = np.arccos((b*b + a*a - c*c) / (2*b*a))

            # Count only defect angles < 90 degrees
            if angle <= np.pi / 2:
                finger_count += 1
                cv2.circle(thresholded_image, far, 4, (255, 0, 0), -1)

    # usually finger count = defects + 1
    hand_obj.fingers = finger_count + 1 if finger_count > 0 else 0

    # VISUALIZATION
    h, w = thresholded_image.shape[:2]
    hull_visual = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(hull_visual, [convexHull], -1, (0, 255, 0), 2)

    cv2.circle(hull_visual, top, 5, (0, 0, 255), -1)
    cv2.circle(hull_visual, bottom, 5, (255, 0, 0), -1)
    cv2.circle(hull_visual, left, 5, (0, 255, 255), -1)
    cv2.circle(hull_visual, right, 5, (255, 255, 0), -1)

    cv2.line(hull_visual, (centerX, 0), (centerX, h), (200, 200, 200), 1)

    cv2.putText(hull_visual, f"Fingers: {hand_obj.fingers}", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow(f"{debug_prefix}Convex Hull", hull_visual)

    return hand_obj



def write_on_image(frame, hand_left, hand_right):
    """Draw labels and both ROI rectangles on the main frame."""
    global frames_elapsed

    # Positions for text
    LEFT_TEXT_POS  = (10, 20)
    RIGHT_TEXT_POS = (10, 50)

    # Left label
    if frames_elapsed < CALIBRATION_TIME:
        text_left = "Calibrating..."
    elif hand_left is None or not hand_left.isInFrame:
        text_left = "Left: No hand"
    else:
        if hand_left.isWaving:
            text_left = "Left: Waving"
        else:
            text_left = f"Left: fingers={hand_left.fingers}"

    # Right label
    if frames_elapsed < CALIBRATION_TIME:
        text_right = "Calibrating..."
    elif hand_right is None or not hand_right.isInFrame:
        text_right = "Right: No hand"
    else:
        if hand_right.isWaving:
            text_right = "Right: Waving"
        else:
            text_right = f"Right: fingers={hand_right.fingers}"

    # Draw text (shadow + white)
    cv2.putText(frame, text_left, LEFT_TEXT_POS,  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),   2, cv2.LINE_AA)
    cv2.putText(frame, text_left, LEFT_TEXT_POS,  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    cv2.putText(frame, text_right, RIGHT_TEXT_POS,  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),   2, cv2.LINE_AA)
    cv2.putText(frame, text_right, RIGHT_TEXT_POS,  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # Draw ROIs on main frame
    cv2.rectangle(frame, (L_region_left, L_region_top), (L_region_right, L_region_bottom), (255,255,255), 2)
    cv2.rectangle(frame, (R_region_left, R_region_top), (R_region_right, R_region_bottom), (255,255,255), 2)


# ---------- Main loop ----------
try:
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Warning: failed to read frame from camera.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)

        # Extract regions and their colored crops
        left_gray, left_color  = get_region(frame, L_region_top, L_region_bottom, L_region_left, L_region_right)
        right_gray, right_color = get_region(frame, R_region_top, R_region_bottom, R_region_left, R_region_right)

        # Initialize accumulators if needed
        if bg_acc_left is None or bg_acc_right is None:
            # use shapes from the first frame
            bg_acc_left  = np.zeros_like(left_gray, dtype=np.float32)
            bg_acc_right = np.zeros_like(right_gray, dtype=np.float32)

        # Calibration phase: accumulate absolute frames
        if frames_elapsed < CALIBRATION_TIME:
            bg_acc_left  = get_average_accumulator(left_gray, bg_acc_left)
            bg_acc_right = get_average_accumulator(right_gray, bg_acc_right)

            # Show calming feedback windows
            cv2.putText(left_color,  f"Calibrating ({frames_elapsed+1}/{CALIBRATION_TIME})", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(right_color, f"Calibrating ({frames_elapsed+1}/{CALIBRATION_TIME})", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow("Left ROI (calib)", left_color)
            cv2.imshow("Right ROI (calib)", right_color)

        # After calibration: finalize backgrounds and start segmentation
        elif frames_elapsed == CALIBRATION_TIME:
            # finalize backgrounds from accumulators
            background_left  = finalize_background_from_acc(bg_acc_left)
            background_right = finalize_background_from_acc(bg_acc_right)
            print("Backgrounds initialized - entering detection mode")

        # else:
        #     # Segment left
        #     L_pair = segment(left_gray, background_left)
        #     if L_pair is not None:
        #         th_L, seg_L = L_pair
        #         # visualize segmentation over colored crop
        #         vis_left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
        #         cv2.drawContours(vis_left, [seg_L], -1, (255, 255, 255), 2)
        #         cv2.imshow("Left Segmented", vis_left)

        #         hand_left = get_hand_data(th_L, seg_L, hand_left, debug_prefix="L_")
        #         if hand_left is not None:
        #             hand_left.isInFrame = True
        #     else:
        #         # nothing detected in left ROI
        #         if hand_left is not None:
        #             hand_left.isInFrame = False
        #         cv2.imshow("Left Segmented", left_color)

        #     # Segment right
        #     R_pair = segment(right_gray, background_right)
        #     if R_pair is not None:
        #         th_R, seg_R = R_pair
        #         vis_right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
        #         cv2.drawContours(vis_right, [seg_R], -1, (255, 255, 255), 2)
        #         cv2.imshow("Right Segmented", vis_right)

        #         hand_right = get_hand_data(th_R, seg_R, hand_right, debug_prefix="R_")
        #         if hand_right is not None:
        #             hand_right.isInFrame = True
        #     else:
        #         if hand_right is not None:
        #             hand_right.isInFrame = False
        #         cv2.imshow("Right Segmented", right_color)

        else:
            # -----------------------------
            # Process LEFT hand
            # -----------------------------
            L_pair = segment(left_gray, background_left)

            if L_pair is not None:
                th_L, seg_L = L_pair

                # --- NEW: apply stronger segmentation ---
                left_color_hsv = cv2.cvtColor(left_color, cv2.COLOR_BGR2HSV)

                # robust skin color range
                lower_skin = np.array([0, 20, 70])
                upper_skin = np.array([20, 255, 255])

                skin_mask = cv2.inRange(left_color_hsv, lower_skin, upper_skin)

                # clean mask
                kernel = np.ones((5, 5), np.uint8)
                skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
                skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
                skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)

                # combine background subtraction + skin mask
                combined_mask = cv2.bitwise_and(th_L, skin_mask)

                # find contour
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    # select the largest region
                    seg_L = max(contours, key=cv2.contourArea)

                    if cv2.contourArea(seg_L) > 3000:     # ignore tiny regions
                        vis_left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
                        cv2.drawContours(vis_left, [seg_L], -1, (255, 255, 255), 2)
                        cv2.imshow("Left Segmented", vis_left)

                        hand_left = get_hand_data(combined_mask, seg_L, hand_left, debug_prefix="L_")
                        if hand_left is not None:
                            hand_left.isInFrame = True
                    else:
                        # too small or noisy
                        if hand_left is not None:
                            hand_left.isInFrame = False
                        cv2.imshow("Left Segmented", left_color)
                else:
                    if hand_left is not None:
                        hand_left.isInFrame = False
                    cv2.imshow("Left Segmented", left_color)

            else:
                if hand_left is not None:
                    hand_left.isInFrame = False
                cv2.imshow("Left Segmented", left_color)



            # -----------------------------
            # Process RIGHT hand
            # -----------------------------
            R_pair = segment(right_gray, background_right)

            if R_pair is not None:
                th_R, seg_R = R_pair

                right_color_hsv = cv2.cvtColor(right_color, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 20, 70])
                upper_skin = np.array([20, 255, 255])
                skin_mask = cv2.inRange(right_color_hsv, lower_skin, upper_skin)

                kernel = np.ones((5, 5), np.uint8)
                skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
                skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
                skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)

                combined_mask = cv2.bitwise_and(th_R, skin_mask)

                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    seg_R = max(contours, key=cv2.contourArea)

                    if cv2.contourArea(seg_R) > 3000:
                        vis_right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
                        cv2.drawContours(vis_right, [seg_R], -1, (255, 255, 255), 2)
                        cv2.imshow("Right Segmented", vis_right)

                        hand_right = get_hand_data(combined_mask, seg_R, hand_right, debug_prefix="R_")
                        if hand_right is not None:
                            hand_right.isInFrame = True
                    else:
                        if hand_right is not None:
                            hand_right.isInFrame = False
                        cv2.imshow("Right Segmented", right_color)
                else:
                    if hand_right is not None:
                            hand_right.isInFrame = False
                    cv2.imshow("Right Segmented", right_color)

            else:
                if hand_right is not None:
                            hand_right.isInFrame = False
                cv2.imshow("Right Segmented", right_color)


        # Write text and ROI rectangles on the main frame
        write_on_image(frame, hand_left, hand_right)

        cv2.imshow("Camera Input", frame)

        frames_elapsed += 1

        # exit
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

finally:
    capture.release()
    cv2.destroyAllWindows()