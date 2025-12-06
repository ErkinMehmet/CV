import cv2
import numpy as np

def show_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# read img
def read_image(image_path):
    image = cv2.imread(image_path)
    print(f"Image shape: {image.shape}")
    show_image("Image", image)

# read video
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: # ret is success flag
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(3) & 0xFF == ord('q'): # press 'q' to quit, cada 3 ms detecta si se ha presionado una tecla
            break
    cap.release()
    cv2.destroyAllWindows()

# read webcam
def read_webcam():
    cap = cv2.VideoCapture(0) # 0 is the default camera
    cap.set(3, 640) # set width
    cap.set(4, 480) # set height
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# convert to grayscale
def to_grayscale(path,showImage=False):
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if showImage:
        show_image("Grayscale Image", gray_image)
    return gray_image

# blur
def blur_image(path, ksize=(7,7),showImage=False):
    image = cv2.imread(path)
    blurred_image = cv2.GaussianBlur(image, ksize, 0)
    if showImage:
        show_image("Blurred Image", blurred_image)
    return blurred_image

# cannyImg
def to_canny(path, threshold1=100, threshold2=200,showImage=False): # from 0 to 255
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, threshold1, threshold2)
    if showImage:
        show_image("Canny Image", canny_image)
    return canny_image

# resize
def resize_image(path, width, height,showImage=False):
    image = cv2.imread(path)
    resized_image = cv2.resize(image, (width, height))
    if showImage:
        show_image("Resized Image", resized_image)
    return resized_image

# crop
def crop_image(path, x, y, w, h,showImage=False):
    image = cv2.imread(path)
    cropped_image = image[y:y+h, x:x+w]
    if showImage:
        show_image("Cropped Image", cropped_image)
    return cropped_image

# line, rectangle, circle, text
def draw_shapes_and_text(path=None,showImage=False):
    if path==None:
        image=np.zeros((512,512,3), np.uint8) # black image with 3 channels and a size of 512x512
    else:
        image = cv2.imread(path)
    # draw line
    cv2.line(image, (50, 50), (200, 50), (255, 0, 0), 5) # blue line 
    # draw rectangle
    cv2.rectangle(image, (50, 100), (200, 200), (0, 255, 0), 3) # green rectangle
    # draw circle
    cv2.circle(image, (300, 150), 50, (0, 0, 255), -1) # red filled circle
    # put text
    cv2.putText(image, "OpenCV", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # font scale 1, white color
    if showImage:
        show_image("Shapes and Text", image)
    return image

# wrap perspective - make a tilted image look flat (rectangle)
def warp_perspective(path, src_points, dst_points=None,showImage=False): # src_points are the corners of the tilted rectangle, dst_points are the corners of the output rectangle
    image = cv2.imread(path)
    if dst_points is None:
        width = max(np.linalg.norm(np.array(src_points[0]) - np.array(src_points[1])),
                    np.linalg.norm(np.array(src_points[2]) - np.array(src_points[3])))
        height = max(np.linalg.norm(np.array(src_points[0]) - np.array(src_points[3])),
                     np.linalg.norm(np.array(src_points[1]) - np.array(src_points[2])))
        dst_points = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    else:
        width=image.shape[1]
        height=image.shape[0]
    matrix = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    warped_image = cv2.warpPerspective(image, matrix, (int(width), int(height)))
    if showImage:
        show_image("Warped Perspective", warped_image)
    return warped_image

# join images
def join_images(path1, path2, axis=1,showImage=False): # axis=1 for horizontal, axis=0 for vertical
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    if axis == 1:
        joined_image = np.hstack((image1, image2))
    else:
        joined_image = np.vstack((image1, image2))
    if showImage:
        show_image("Joined Images", joined_image)
    return joined_image

