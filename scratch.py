import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as matpltimg
import glob


# real-time canny edge detection based on webcam image

def nothing(x):
    pass



cam = cv.VideoCapture(0)
if cam is None or not cam.isOpened():
    print('Warning: unable to open video source')
    exit(1)

cam.set(3, 640)  # width and height
cam.set(4, 480)

cv.namedWindow('trackbars')
cv.createTrackbar('threshold_low', 'trackbars', 0, 255, nothing)
cv.createTrackbar('threshold_high', 'trackbars', 0, 255, nothing)

while True:
    try:
        success, frame = cam.read()
    except:
        print("Cam not available.")
        break

    #  write program here

    threshold_low = cv.getTrackbarPos('threshold_low', 'trackbars')
    threshold_high = cv.getTrackbarPos('threshold_high', 'trackbars')

    canny = cv.Canny(frame, threshold_low, threshold_high)
    cv.imshow('WEBCAM', frame)
    cv.imshow('CANNY', canny)

    # end program here

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()

# histograms


img = cv.imread('photo.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

color = ('r', 'g', 'b')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

ax1.imshow(img)

ax2.hist(img.ravel(), 256, [0, 256], color="grey")

for i in enumerate(color):
    color_histogram = cv.calcHist([img], [i[0]], None, [256], [0, 256])
    ax3.plot(color_histogram, color=i[1])

plt.show()


# canny edge detection


def resize_img(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_img = cv.resize(frame, (width, height))
    return resized_img


def nothing(x):
    pass


cv.namedWindow('trackbars')
cv.createTrackbar('threshold_low', 'trackbars', 0, 255, nothing)
cv.createTrackbar('threshold_high', 'trackbars', 255, 255, nothing)

while True:

    #  write program here

    threshold_low = cv.getTrackbarPos('threshold_low', 'trackbars')
    threshold_high = cv.getTrackbarPos('threshold_high', 'trackbars')

    resized_img = resize_img(img, 30)
    resized_img = cv.cvtColor(resized_img, cv.COLOR_RGB2BGR)

    canny = cv.Canny(resized_img, threshold_low, threshold_high)
    cv.imshow('PHOTO', resized_img)
    cv.imshow('CANNY', canny)

    # end program here

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()

# high and low pass filters


img = matpltimg.imread('photo.jpg')

mask_low_pass = np.array([[1, 1, 1],
                          [1, 5, 1],
                          [1, 1, 1]], dtype=float)

mask_low_pass = mask_low_pass / np.sum(mask_low_pass)

mask_high_pass = np.array([[0.0, -2.0, 0.0],
                           [-2.0, 10.0, -2.0],
                           [0.0, -2.0, 0.0]], dtype=float)

mask_high_pass = mask_high_pass / (np.sum(mask_high_pass) if np.sum(mask_high_pass) != 0 else 1)

output_low_pass_img = cv.filter2D(img, -1, mask_low_pass)
output_high_pass_img = cv.filter2D(img, -1, mask_high_pass)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
ax1.imshow(img)
ax1.set_title("Original")
ax2.imshow(output_low_pass_img)
ax2.set_title("Low pass filter")
ax3.imshow(output_high_pass_img)
ax3.set_title("High pass filter")
plt.show()
