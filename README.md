# Emojify — Real-Time Facial Expression

Turn live **facial expressions** into matching **emojis** on your webcam feed. Train a 7-class CNN on face images and run a real-time overlay that places the predicted emoji next to each detected face.

> **Use cases:** fun overlays for streams, video chat reactions, quick demos of CV + deep learning, and accessibility/education.

---

## ✨ What’s in this project

- **Live webcam app** that detects faces (OpenCV Haar cascade), classifies emotion with a Keras model, and overlays the corresponding emoji image on the frame in real time. :contentReference[oaicite:0]{index=0}  
- **CNN training script** that builds and trains a 7-class emotion classifier on 48×48 face images using Keras, then saves a `.h5` model. :contentReference[oaicite:1]{index=1}  
- **Emoji assets** expected in an `emojis/` folder (e.g., `happy.png`, `sad.png`, …) used by the overlay app. :contentReference[oaicite:2]{index=2}

---

## 🧱 Model & labels

- **Architecture (train.py):** Conv2D + MaxPooling + Dropout blocks → Flatten → Dense(1024) → Dropout → Dense(7, softmax). :contentReference[oaicite:5]{index=5}  
- **Classes (7):** `Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral`. Order is used at inference to map model outputs → emoji files. :contentReference[oaicite:6]{index=6}  
- **Training images:** 48×48 faces; script config uses `ImageDataGenerator(..., target_size=(48,48))`. :contentReference[oaicite:7]{index=7}  
- **Inference (gui.py):** crops a face ROI, resizes to **48×48 grayscale**, scales to `[0,1]`, and predicts. :contentReference[oaicite:8]{index=8}

> **Heads-up:** `train.py` builds the model with `input_shape=(48,48,1)` (grayscale) but loads data with `color_mode='rgb'`. Either switch to `color_mode='grayscale'` in `train.py` **or** change the model input shape to `(48,48,3)` to match. :contentReference[oaicite:9]{index=9}


