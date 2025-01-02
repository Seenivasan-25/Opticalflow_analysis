import cv2 as cv
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(41)

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(flow_gt, flow_pred):
    return np.mean((flow_gt - flow_pred)**2)

# Save results to disk for inclusion in the paper
def save_results(frame,output_dir,frame_count):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    cv.imwrite(output_path, frame)

# Lucas-Kanade Optical Flow (Sparse Method)
def lucasKanade(output_dir="lucas_kanade_results"):
    root = os.getcwd()
    videoPath = os.path.join(root, 'mixkit-one-on-one-in-a-soccer-game-43483-hd-ready.mp4')
    videoCapObj = cv.VideoCapture(videoPath)

    # Parameters for Shi-Tomasi corner detection
    shiTomasiCornerParams = dict(maxCorners=20, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lucasKanadeParams = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS, 10, 0.03))

    # Random colors for drawing
    randomColors = np.random.randint(0, 255, (100, 3))

    # Read the first frame
    _, frameFirst = videoCapObj.read()
    frameGrayPrev = cv.cvtColor(frameFirst, cv.COLOR_BGR2GRAY)

    # Detect good features to track (Shi-Tomasi corner detection)
    cornersPrev = cv.goodFeaturesToTrack(frameGrayPrev, mask=None, **shiTomasiCornerParams)

    # Create a mask image for drawing tracks
    mask = np.zeros_like(frameFirst)

    frame_count = 0
    while True:
        ret, frame = videoCapObj.read()
        if not ret:
            break

        frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculate optical flow (Lucas-Kanade method)
        cornersCur, foundStatus, _ = cv.calcOpticalFlowPyrLK(frameGrayPrev, frameGray, cornersPrev, None,
                                                             **lucasKanadeParams)

        if cornersCur is not None:
            cornersMatchesCur = cornersCur[foundStatus == 1]
            cornersMatchesPrev = cornersPrev[foundStatus == 1]

            for i, (curCorner, prevCorner) in enumerate(zip(cornersMatchesCur, cornersMatchesPrev)):
                xCur, yCur = curCorner.ravel()
                xPrev, yPrev = prevCorner.ravel()


                # Draw lines and circles on the mask and frames                mask = cv.line(mask, (int(xCur), int(yCur)), (int(xPrev), int(yPrev)), randomColors[i].tolist(), 2)
                frame = cv.circle(frame, (int(xCur), int(yCur)), 5, randomColors[i].tolist(), -1)
            # Overlay the mask on the frame
            img = cv.add(frame, mask)
            save_results(img, output_dir, frame_count)

            cv.imshow('Lucas-Kanade Optical Flow', img)

        # Update the previous frame and corners
        frameGrayPrev = frameGray.copy()
        cornersPrev = cornersMatchesCur.reshape(-1, 1, 2)

        frame_count += 1
        if cv.waitKey(15) & 0xFF == ord('q'):
            break

    videoCapObj.release()
    cv.destroyAllWindows()

# Dense Optical Flow (Farneback method)
def denseOpticalFlow(output_dir="dense_optical_flow_results"):
    root = os.getcwd()
    videoPath = os.path.join(root, 'mixkit-one-on-one-in-a-soccer-game-43483-hd-ready.mp4')

    videoCapObj = cv.VideoCapture(videoPath)

    # Read the first frame and convert to grayscale
    _, frameFirst = videoCapObj.read()
    imgPrev = cv.cvtColor(frameFirst, cv.COLOR_BGR2GRAY)

    # Initialize HSV image for visualizing the flow
    imgHSV = cv.cvtColor(frameFirst, cv.COLOR_BGR2HSV)
    imgHSV[:, :, 1] = 255  # Set saturation to maximum

    frame_count = 0
    while True:
        ret, frameCur = videoCapObj.read()
        if not ret:
            break

        imgCur = cv.cvtColor(frameCur, cv.COLOR_BGR2GRAY)

        # Compute dense optical flow using Farneback method
        flow = cv.calcOpticalFlowFarneback(imgPrev, imgCur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert flow to polar coordinates (magnitude and angle)
        mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1])

        # Set the angle and magnitude in the HSV image
        imgHSV[:, :, 0] = ang * 180 / np.pi / 2
        imgHSV[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # Convert HSV to BGR
        imgBGR = cv.cvtColor(imgHSV, cv.COLOR_HSV2BGR)
        save_results(imgBGR, output_dir, frame_count)

        cv.imshow('Dense Optical Flow', imgBGR)

        imgPrev = imgCur.copy()

        frame_count += 1
        if cv.waitKey(15) & 0xFF == ord('q'):
            break

    videoCapObj.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    # Run both methods and save results for further analysis
    lucasKanade()
    denseOpticalFlow()