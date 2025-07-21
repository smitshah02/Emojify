import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# Load pre-trained facial expression classification model
model = load_model('face_predict.h5')

# Define a function to generate emojis based on facial expressions
def generate_emoji(image, model):
    # Preprocess the image
    image = cv2.resize(image, (48, 48))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(1, 48, 48, 1)
    image = image / 255.0

    # Use the model to classify the facial expression
    prediction = model.predict(image)
    emoji = np.argmax(prediction)
    emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
    emotion = emotions[emoji]

    # Generate an emoji based on the classification
    if emoji == 0:
        emoji_image = Image.open('emojis/angry.png')
    elif emoji == 1:
        emoji_image = Image.open('emojis/disgusted.png')
    elif emoji == 2:
        emoji_image = Image.open('emojis/fearful.png')
    elif emoji == 3:
        emoji_image = Image.open('emojis/happy.png')
    elif emoji == 4:
        emoji_image = Image.open('emojis/sad.png')
    elif emoji == 5:
        emoji_image = Image.open('emojis/surprised.png')
    else:
        emoji_image = Image.open('emojis/neutral.png')

    draw = ImageDraw.Draw(emoji_image)
    font = ImageFont.truetype('arial.ttf', 30)
    draw.text((10, 10), emotion, font=font)

    # Return the emoji image
    return emoji_image

# Define a function to capture video from webcam and generate emojis in real-time
def capture_video():
    # Set up video capture device
    cap = cv2.VideoCapture(0)

    # Set output frame size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width+150, frame_height))

    while True:
        # Capture frame from video feed
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Create a blank output frame
        output_frame = np.zeros((frame_height, frame_width+150, 3), np.uint8)

        # Copy the original frame to the output frame
        output_frame[0:frame_height, 0:frame_width] = frame

        # For each detected face, generate an appropriate emoji and overlay it on the output frame
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            emoji_image = generate_emoji(face_image, model)
            emoji_image = emoji_image.resize((h, h))
            #output_frame[y:y+h//2, x+w:x+w+(emoji_image.size[0])] = cv2.cvtColor(np.array(emoji_image), cv2.COLOR_RGB2BGR)
            output_frame[y:y+h, x+w:x+2*w] = cv2.cvtColor(np.array(emoji_image), cv2.COLOR_RGB2BGR)

        # Write the output frame to the video file
        out.write(output_frame)

        # Display the output frame
        cv2.imshow('Emojiify', output_frame)

        # Exit on 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

# Call the capture_video() function to start generating emojis
capture_video()
