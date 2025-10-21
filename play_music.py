# import spotipy
# from spotipy.oauth2 import SpotifyOAuth

# # Authentication manager
# sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
#     client_id="7d6eaf01b0a04f759066945e26c636b5",
#     client_secret="616a28cc686949a3a76e073744b5fa1f",
#     redirect_uri="http://127.0.0.1:8029/callback",
#     scope="user-modify-playback-state user-read-playback-state"
# ))

# # Example song (Rick Astley 😆)
# track_uri = "spotify:track:4cOdK2wGLETKBW3PvgPWqT"

# # Try to play
# devices = sp.devices()

# if devices['devices']:
#     # Play on first available device
#     device_id = devices['devices'][0]['id']
#     sp.start_playback(device_id=device_id, uris=[track_uri])
#     print("Playing song on Spotify...")
# else:
#     print("No active Spotify device found. Open Spotify app first.")

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Spotify Setup ---
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    redirect_uri="http://localhost:8888/callback",
    scope="user-read-playback-state user-modify-playback-state"
))

playlist_uri = "spotify:playlist:YOUR_PLAYLIST_ID"

# Keep track of playback state
is_playing_music = False

def play_music():
    global is_playing_music
    playback = sp.current_playback()

    if not playback or not playback.get("is_playing", False):
        # Only start playlist if nothing is playing
        sp.shuffle(True)  # randomize tracks
        sp.start_playback(context_uri=playlist_uri)
        is_playing_music = True
        print("▶️ Music started (shuffled).")
    else:
        print("🎶 Music already playing, no restart.")

def stop_music():
    global is_playing_music
    playback = sp.current_playback()

    if playback and playback.get("is_playing", False):
        sp.pause_playback()
        is_playing_music = False
        print("⏸️ Music paused.")

