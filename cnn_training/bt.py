import cv2


def mask_hand(img):
    # Extract red color channel (because the hand color is more red than the background).
    gray = img[:, :, 1:2]
    cv2.imshow('',gray)

    # Apply binary threshold using automatically selected threshold (using cv2.THRESH_OTSU parameter).
    ret, thresh_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use "opening" morphological operation for clearing some small dots (noise)
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

    # Use "closing" morphological operation for closing small gaps
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))

    # Display result:
    cv2.imshow('thresh_gray', cv2.resize(thresh_gray, (thresh_gray.shape[1]//2, thresh_gray.shape[0]//2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('./ASL_Dataset/Train/A/5.jpg')