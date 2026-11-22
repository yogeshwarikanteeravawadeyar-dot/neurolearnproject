import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np
import pickle
import threading
import time
import pygame
from gtts import gTTS
from mutagen.mp3 import MP3

class GestureDetector:
    def __init__(self):
        # Load model
        try:
            with open('model.pkl', 'rb') as f:
                self.svm = pickle.load(f)
        except FileNotFoundError:
            print("Warning: model.pkl not found. Using dummy classifier.")
            from sklearn.ensemble import RandomForestClassifier
            self.svm = RandomForestClassifier()
            # Create dummy model for demonstration
            dummy_X = np.random.rand(100, 63)
            dummy_y = np.random.choice(['A', 'B', 'C'], 100)
            self.svm.fit(dummy_X, dummy_y)
        
        # MediaPipe setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
        # Camera setup
        self.cap = None
        self.is_running = False
        
        # Gesture tracking
        self.current_gesture = ""
        self.gesture_history = []
        self.current_word = ""
        self.callback_function = None
        
        # Demo image
        self.demo_image = None
        try:
            self.demo_image = cv2.imread('demo.JPG')
            if self.demo_image is not None:
                self.demo_image = cv2.resize(self.demo_image, (500, 500))
        except:
            print("Demo image not found, creating placeholder")
            self.demo_image = np.zeros((500, 500, 3), dtype=np.uint8)
            cv2.putText(self.demo_image, "ASL Reference", (150, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def extract_landmarks(self, hand_landmarks):
        """Extract clean landmark data from hand landmarks"""
        without_garbage = []
        clean = []
        
        data = str(hand_landmarks)
        data = data.strip().split('\n')
        
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        
        for item in data:
            if item not in garbage:
                without_garbage.append(item)
        
        for item in without_garbage:
            item = item.strip()
            clean.append(item[2:])
        
        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
            
        return clean
##    
##    def speak_text(self, text):
##        """Text to speech function"""
##        try:
##            myobj = gTTS(text=text, lang='en-us', tld='com', slow=False)
##            filename = f"static/voice_files/gesture_voice_{int(time.time())}.mp3"
##            myobj.save(filename)
##            
##            pygame.mixer.init()
##            pygame.mixer.music.load(filename)
##            pygame.mixer.music.play()
##            
##            song = MP3(filename)
##            time.sleep(song.info.length)
##            pygame.mixer.quit()
##            
##            # Clean up file after playing
##            if os.path.exists(filename):
##                os.remove(filename)
##        except Exception as e:
##            print(f"Error in gesture speech: {e}")
    
    def start_detection(self, callback=None):
        """Start the gesture detection"""
        self.is_running = True
        self.callback_function = callback
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop the gesture detection"""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    def clear_word(self):
        """Clear the current word"""
        self.current_word = ""
        if self.callback_function:
            self.callback_function({
                'type': 'word_cleared',
                'timestamp': time.time()
            })

    def speak_current_word(self):
        """Speak the current word"""
        if self.current_word:
            self.speak_text(f"Spelling: {self.current_word}")
            if self.callback_function:
                self.callback_function({
                    'type': 'word_spoken',
                    'word': self.current_word,
                    'timestamp': time.time()
                })

    def speak_text(self, text):
        """Text to speech function"""
        try:
            myobj = gTTS(text=text, lang='en-us', tld='com', slow=False)
            filename = f"static/voice_files/gesture_voice_{int(time.time())}.mp3"
            myobj.save(filename)
            
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            song = MP3(filename)
            time.sleep(song.info.length)
            pygame.mixer.quit()
            
            # Clean up file after playing
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f"Error in gesture speech: {e}")    
    def _detection_loop(self):
        """Main detection loop"""
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        with self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            
            last_gesture_time = 0
            gesture_cooldown = 2  # seconds between gesture recognition
            
            while self.is_running:
                try:
                    success, image = self.cap.read()
                    if not success:
                        continue

                    # Resize image
                    image = cv2.resize(image, (500, 500))
                    
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)

                    # Draw the hand annotations on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    current_gesture = ""
                    if results.multi_hand_landmarks:
                        n = len(results.multi_hand_landmarks)
                        if n == 1:  # Only process single hand gestures
                            for hand_landmarks in results.multi_hand_landmarks:
                                self.mp_drawing.draw_landmarks(
                                    image,
                                    hand_landmarks,
                                    self.mp_hands.HAND_CONNECTIONS,
                                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                    self.mp_drawing_styles.get_default_hand_connections_style())
                            
                            # Extract and process landmarks
                            clean_landmarks = self.extract_landmarks(hand_landmarks)
                            
                            if len(clean_landmarks) == 63:  # Ensure we have all landmarks
                                # Predict gesture
                                gesture = self.svm.predict(np.array(clean_landmarks).reshape(-1, 63))
                                current_gesture = gesture[0]
                                
                                # Only process new gestures after cooldown
                                current_time = time.time()
                                if (current_time - last_gesture_time > gesture_cooldown and 
                                    current_gesture != self.current_gesture):
                                    
                                    self.current_gesture = current_gesture
                                    last_gesture_time = current_time
                                    
                                    # Add to history
                                    self.gesture_history.append(current_gesture)
                                    if len(self.gesture_history) > 10:
                                        self.gesture_history.pop(0)
                                    
                                    # Add to current word
                                    self.current_word += current_gesture
                                    
                                    # Speak the detected letter
                                    self.speak_text(current_gesture)
                                    
                                    # Call callback function
                                    if self.callback_function:
                                        self.callback_function({
                                            'type': 'gesture_detected',
                                            'gesture': current_gesture,
                                            'current_word': self.current_word,
                                            'timestamp': current_time
                                        })
                                    
                                    print(f"Gesture detected: {current_gesture}")
                                
                                # Display gesture on image
                                cv2.putText(image, f"ASL: {current_gesture}", (50, 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Display current word
                    cv2.putText(image, f"Word: {self.current_word}", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Combine with demo image
                    if self.demo_image is not None:
                        final_frame = cv2.hconcat([image, self.demo_image])
                    else:
                        final_frame = image
                    
                    # Display the frame
                    cv2.imshow('Gesture Recognition - ASL to Voice', final_frame)
                    
                    # Check for quit command
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                    elif key == ord('c'):  # Clear word
                        self.current_word = ""
                    elif key == ord('s'):  # Speak word
                        if self.current_word:
                            self.speak_text(f"Word: {self.current_word}")
                            
                except Exception as e:
                    print(f"Error in gesture detection loop: {e}")
                    continue
        
        # Cleanup
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    
    def clear_word(self):
        """Clear the current word"""
        self.current_word = ""
        if self.callback_function:
            self.callback_function({
                'type': 'word_cleared',
                'timestamp': time.time()
            })
    
    def speak_current_word(self):
        """Speak the current word"""
        if self.current_word:
            self.speak_text(f"Spelling: {self.current_word}")
            if self.callback_function:
                self.callback_function({
                    'type': 'word_spoken',
                    'word': self.current_word,
                    'timestamp': time.time()
                })
    
    def get_status(self):
        """Get current status of the detector"""
        return {
            'is_running': self.is_running,
            'current_gesture': self.current_gesture,
            'current_word': self.current_word,
            'gesture_history': self.gesture_history,
            'word_length': len(self.current_word)
        }

# Global instance
gesture_detector = GestureDetector()
