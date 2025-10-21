# Mood-Based-Music-Player-with-Gesture-Control

An AI + Computer Vision project that detects human emotions through facial expressions and controls Spotify playlists automatically. Added an extra twist — hand gesture control ✋👊 to pause or resume mood detection in real-time.

🧠 Overview

This project integrates emotion recognition, gesture detection, and Spotify API control to create a personalized music experience.
The system uses a webcam to:

Detect a user’s facial emotion (Happy, Sad, Angry, Neutral)

Play a Spotify playlist mapped to that emotion

Allow gesture-based control (pause/resume detection)

🔧 Tech Stack

Programming Language: Python 🐍

Libraries Used:

OpenCV → Real-time image capture & face detection

TensorFlow / Keras → Emotion classification model (emotion_model.h5)

⚙️ Features

✅ Real-time emotion detection using webcam
✅ Dynamic Spotify playlist mapping
✅ Hand gestures to pause/resume detection
✅ Shuffled playback for natural experience
✅ Works with any Spotify device (Desktop/Mobile/Web Player)

🚀 How It Works

*Facial Emotion Detection

*The webcam captures live frames

*The model predicts emotion from detected face

*Spotify Integration

*Each emotion is linked to a specific Spotify playlist URI

*The app controls playback via the Spotify API

#Gesture Control

✋ Open palm → Pause emotion-based detection

👊 Fist → Resume detection and playlist switching

🧩 Setup Instructions

#Clone the Repository:
{
git clone https://github.com/yourusername/mood-music-gesture.git
cd mood-music-gesture
}

#Create and Activate Virtual Environment:
{
python -m venv moodenv
moodenv\Scripts\activate      # Windows
source moodenv/bin/activate   # macOS/Linux
}

#Install Dependencies:
{
pip install -r requirements.txt
}

##Spotify API Setup:

1)Go to Spotify Developer Dashboard

2)Create an app → Get your Client ID and Client Secret

3)Set Redirect URI as http://127.0.0.1:8029/callback

4)Add these credentials inside the code or .env file

5)Run the Project:

6)python app.py


NOTE--Make Sure:

Spotify app is open on your device

You have a Spotify Premium account for playback control


💡 Challenges Faced

Compatibility issues with TensorFlow and Mediapipe on newer Python versions

Managing Spotify device connection during playback

And yes, upgrading to Spotify Premium was part of the debugging journey 😅

📚 Future Improvements

Add support for more emotions (e.g., Fear, Surprise)

Enable multi-user detection

Integrate local music playback for non-Premium users
