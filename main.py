# Import essential libraries
import requests
import matplotlib.pylab as plt
import cv2
import numpy as np
import imutils


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(image, lines):
    img = np.copy(image)
    blank_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 1)

    return img

# To capture video from Url.
url = "http://192.168.0.192:8080/shot.jpg"

# Resolution for output video
frame_width = 1280
frame_height = 720

out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
out2 = cv2.VideoWriter('output2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height), 0)

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    image = imutils.resize(img, width=frame_width, height=frame_height)

    #print(image.shape)

    height = image.shape[0]
    width = image.shape[1]

    print(height, width)

    region_of_interest_vertices = [
        (width*2/8, height),
        (width*3.2/8, height*6.5/8),
        (width*5.5/8, height*6.5/8),
        (width*6.5/8, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 195, 200, cv2.THRESH_BINARY)

    cropped_image = region_of_interest(blackAndWhiteImage, np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=10,
                            lines=np.array([]),
                            minLineLength=20,
                            maxLineGap=5)

    image_with_lines = draw_the_lines(image, lines)

    # Write video output
    out.write(image_with_lines)
    out2.write(blackAndWhiteImage)
    # Display
    cv2.imshow('blackAndWhiteImage', blackAndWhiteImage)
    cv2.imshow('output',image_with_lines)
    #plt.imshow(cropped_image)
    #plt.show()

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k== ord('q'):
        break

# Release the VideoCapture object
out2.release()
out.release()

cv2.destroyAllWindows()