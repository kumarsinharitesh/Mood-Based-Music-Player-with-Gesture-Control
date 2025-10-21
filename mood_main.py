# import cv2
# import numpy as np
# import tensorflow as tf
# import time
# import spotipy
# from spotipy.oauth2 import SpotifyOAuth

# # -------------------------------
# # 1. Load trained model
# # -------------------------------
# model = tf.keras.models.load_model("emotion_model.h5")
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # -------------------------------
# # 2. Spotify Setup
# # -------------------------------
# sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
#     client_id="7d6eaf01b0a04f759066945e26c636b5",
#     client_secret="616a28cc686949a3a76e073744b5fa1f",
#     redirect_uri="http://127.0.0.1:8029/callback",
#     scope="user-modify-playback-state user-read-playback-state"
# ))

# devices = sp.devices()
# if not devices['devices']:
#     print("⚠ No active Spotify device found. Please open Spotify app first.")
#     exit()

# device_id = devices['devices'][0]['id']

# # Map emotions to playlist URIs
# mood_to_playlist = {
#     "Happy": "spotify:playlist:1nrr4tJtJIOaiRJH9whcX8",
#     "Sad": "spotify:playlist:6iKLu5E6ECy8BG7IsQ2HA8",
#     "Angry": "spotify:playlist:7zHDEtnqHpXYYSYMfmHB6q",
#     "Neutral": "spotify:playlist:31jLEAx7h0OrgogbHQvXbJ"
#     # Add more if you want
# }

# # -------------------------------
# # 3. Face Detection Setup
# # -------------------------------
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# cap = cv2.VideoCapture(0)

# last_emotion = None
# last_time = 0

# # -------------------------------
# # 4. Loop for real-time detection
# # -------------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_resized = cv2.resize(roi_gray, (48, 48))
#         roi_normalized = roi_resized.astype("float") / 255.0
#         roi_reshaped = np.expand_dims(roi_normalized, axis=-1)
#         roi_reshaped = np.expand_dims(roi_reshaped, axis=0)
        
#         predictions = model.predict(roi_reshaped, verbose=0)
#         emotion_idx = np.argmax(predictions)
#         emotion = emotion_labels[emotion_idx]
        
#         cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
#                     1, (0,255,0), 2, cv2.LINE_AA)

#         # --- Spotify Playback Logic ---
#         if emotion in mood_to_playlist and (time.time() - last_time > 15):
#             playlist_uri = mood_to_playlist[emotion]

#             # Get current playback info
#             playback = sp.current_playback()

#             # Play only if new emotion OR nothing playing
#             if (emotion != last_emotion) or (not playback or not playback.get("is_playing", False)):
#                 sp.shuffle(True, device_id=device_id)  # shuffle ON
#                 sp.start_playback(device_id=device_id, context_uri=playlist_uri)
#                 print(f"🎶 Playing {emotion} playlist on Spotify (shuffled)")

#                 last_emotion = emotion
#                 last_time = time.time()
#             else:
#                 print(f"✅ Already playing {emotion} playlist, not restarting.")
    
#     cv2.imshow("Emotion → Spotify", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import mediapipe as mp

# -------------------------------
# 1. Load trained model
# -------------------------------
model = tf.keras.models.load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -------------------------------
# 2. Spotify Setup
# -------------------------------
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="7d6eaf01b0a04f759066945e26c636b5",
    client_secret="616a28cc686949a3a76e073744b5fa1f",
    redirect_uri="http://127.0.0.1:8029/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

devices = sp.devices()
if not devices['devices']:
    print("⚠ No active Spotify device found. Please open Spotify app first.")
    exit()

device_id = devices['devices'][0]['id']

# Map emotions to playlist URIs
mood_to_playlist = {
    "Happy": "spotify:playlist:1nrr4tJtJIOaiRJH9whcX8",
    "Sad": "spotify:playlist:6iKLu5E6ECy8BG7IsQ2HA8",
    "Angry": "spotify:playlist:7zHDEtnqHpXYYSYMfmHB6q",
    "Neutral": "spotify:playlist:31jLEAx7h0OrgogbHQvXbJ"
    # Add more if you want
}

# -------------------------------
# 3. Face Detection Setup
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

last_emotion = None
last_time = 0

# -------------------------------
# 4. Hand Detection Setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
detection_paused = False

# -------------------------------
# 5. Loop for real-time detection
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural mirror interaction
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Hand Gesture Detection ---
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            finger_tips = [8, 12, 16, 20]
            thumb_tip = 4
            fingers = []

            # Thumb check
            if landmarks[thumb_tip].x < landmarks[thumb_tip - 2].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            for tip in finger_tips:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

            # ✋ Open palm → Pause detection
            if total_fingers == 5:
                detection_paused = True
                cv2.putText(frame, "✋ Detection Paused", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 👊 Fist → Resume detection
            elif total_fingers == 0:
                detection_paused = False
                cv2.putText(frame, "👊 Detection Active", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- Emotion Detection & Spotify only if not paused ---
    if not detection_paused:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized.astype("float") / 255.0
            roi_reshaped = np.expand_dims(roi_normalized, axis=-1)
            roi_reshaped = np.expand_dims(roi_reshaped, axis=0)

            predictions = model.predict(roi_reshaped, verbose=0)
            emotion_idx = np.argmax(predictions)
            emotion = emotion_labels[emotion_idx]

            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # --- Spotify Playback Logic ---
            if emotion in mood_to_playlist and (time.time() - last_time > 15):
                playlist_uri = mood_to_playlist[emotion]

                playback = sp.current_playback()

                if (emotion != last_emotion) or (not playback or not playback.get("is_playing", False)):
                    sp.shuffle(True, device_id=device_id)
                    sp.start_playback(device_id=device_id, context_uri=playlist_uri)
                    print(f"🎶 Playing {emotion} playlist on Spotify (shuffled)")

                    last_emotion = emotion
                    last_time = time.time()
                else:
                    print(f"✅ Already playing {emotion} playlist, not restarting.")

    cv2.imshow("Emotion → Spotify", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
