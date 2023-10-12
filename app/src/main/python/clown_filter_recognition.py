import numpy as np
import cv2
import io
from PIL import Image
import base64
from os.path import dirname, join

noseCascade = ""


def __init__():
    filename = join(dirname(__file__), "haarcascade_mcs_nose_prova.xml")
    filename=dirname(cv2.__file__) + "/data/haarcascade_mcs_nose_prova.xml"
    print(filename)
    noseCascade = cv2.CascadeClassifier(filename)
    if noseCascade.empty():
        raise IOError('Unable to load the nose cascade classifier xml file')



def noseDetection(image):
    decoded_data = base64.b64decode(image)
    np_data = np.fromstring(decoded_data, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nose = noseCascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in nose:
        # Draw a rectangle around the face
        color = (255, 0, 0)  # in BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
    pil_img = Image.fromarray(frame)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    img_str = base64.b64decode(buff.getvalue())
    return "" + str(img_str, "utf-8")


""""
def detection(input):
    # input è string image
    decoded_data = base64.b64decode(input)
    np_data = np.fromstring(decoded_data, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        roi_rgb = rgb[y:y + h, x:x + w]
        color = (255, 0, 0)  # in BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # resize the image
        size = (224, 224)
        resized_image = cv2.resize(roi_rgb, size)
        image_array = np.array(resized_image, "uint8")
        img = image_array.reshape(1, image_width, image_height, 3)
        img = img.astype('float32')
        img /= 255

        # img_array = keras.utils.img_to_array(img)
        # img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img)
        score = tf.nn.softmax(predictions[0])

        # Display the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[np.argmax(score)]
        color = (255, 0, 255)
        stroke = 2
        cv2.putText(frame, '{} {:.2f}%'.format(name, 100 * np.max(score)), (x, y - 8),
                    font, 1, color, stroke, cv2.LINE_AA)
        # Get faces into webcam's image
        rects = detector(gray, 0)
        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            tl = shape[31][0], shape[29][1]
            tr = shape[35][0], shape[29][1]
            bl = shape[31][0], shape[33][1]
            br = shape[35][0], shape[33][1]

            # dimensioni del naso
            nose_width = tr[0] - tl[0]
            nose_height = bl[1] - tl[1]
            nose = cv2.resize(nose_png, (int(nose_width), int(nose_height)))

            nose_area = frame[int(tl[1]):int(br[1]), int(tl[0]):int(br[0])]

            # per toglie sfondo nero del naso
            nose_gray = cv2.cvtColor(nose, cv2.COLOR_BGR2GRAY)
            r1, nose_mask = cv2.threshold(
                nose_gray, 25, 255, cv2.THRESH_BINARY_INV)
            nose_area_2 = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)

            # attacco tutto
            if (name == 'Steve'):
                nose_final = cv2.add(nose_area_2, nose)
                frame[int(tl[1]):int(br[1]), int(tl[0]):int(br[0])] = nose_final
"""
