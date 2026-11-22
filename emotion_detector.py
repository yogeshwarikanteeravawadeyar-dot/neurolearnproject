import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime
import logging
from flask import current_app
from flask_sqlalchemy import SQLAlchemy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotionDetector:
    def __init__(self, db=None, user_id=None):
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                            3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        
        # Database and user context
        self.db = db
        self.user_id = user_id  # Store user ID instead of relying on current_user
        
        # Initialize models
        self.emotion_model = None
        self.face_detector = None
        self._load_models()
        
        # Camera and detection setup
        self.cap = None
        self.is_running = False
        
        # Emotion tracking
        self.current_emotion = "Neutral"
        self.emotion_history = []
        self.emotion_stats = {emotion: 0 for emotion in self.emotion_dict.values()}
        self.callback_function = None
        self.last_save_time = time.time()
        self.pending_saves = []  # Store emotions to save to database
        
        logging.info("EmotionDetector initialized")

    def set_user_context(self, db, user_id):
        """Set database and user context for saving data"""
        self.db = db
        self.user_id = user_id
        logging.info(f"User context set for user_id: {user_id}")

    def _load_models(self):
        """Load emotion model and face detector exactly like the reference code"""
        try:
            # Load emotion model - EXACTLY like reference code
            json_file = open('model/emotion_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            from keras.models import model_from_json
            self.emotion_model = model_from_json(loaded_model_json)
            self.emotion_model.load_weights("model/emotion_model.h5")
            logging.info("Loaded emotion model from disk")
            
        except Exception as e:
            logging.error(f"Error loading emotion model: {e}")
            # Create a simple fallback model for basic testing
            self._create_fallback_model()
        
        try:
            # Load face detector - EXACTLY like reference code
            self.face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            if self.face_detector.empty():
                logging.error("Face detector cascade file is empty or invalid")
                self.face_detector = None
            else:
                logging.info("Loaded face detector cascade")
                
        except Exception as e:
            logging.error(f"Error loading face detector: {e}")
            self.face_detector = None

    def _create_fallback_model(self):
        """Create a simple fallback model for testing"""
        try:
            from keras.models import Sequential
            from keras.layers import Dense, Conv2D, Flatten
            
            # Create a simple CNN model architecture similar to emotion model
            self.emotion_model = Sequential([
                Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
                Conv2D(64, (3,3), activation='relu'),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(7, activation='softmax')
            ])
            
            # Use random weights for fallback (will give random predictions)
            self.emotion_model.compile(optimizer='adam', loss='categorical_crossentropy')
            logging.info("Created fallback emotion model with random weights")
            
        except Exception as e:
            logging.error(f"Error creating fallback model: {e}")
            self.emotion_model = None

    def _initialize_camera(self):
        """Initialize camera exactly like reference code"""
        try:
            # Start the webcam feed - EXACTLY like reference code
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                logging.info("Camera initialized successfully")
                return cap
            else:
                logging.error("Could not open camera")
                return None
                
        except Exception as e:
            logging.error(f"Error initializing camera: {e}")
            return None

    def _save_emotion_to_db(self, emotion, duration):
        """Save emotion data to database"""
        try:
            if self.db and self.user_id:
                from datetime import datetime
                
                # Import the EmotionData model here to avoid circular imports
                from your_main_app_file import EmotionData  # Replace with your actual main app file name
                
                emotion_data = EmotionData(
                    user_id=self.user_id,
                    emotion=emotion,
                    duration=duration,
                    recorded_at=datetime.now()
                )
                
                self.db.session.add(emotion_data)
                self.db.session.commit()
                logging.info(f"Saved emotion '{emotion}' to database for user {self.user_id}")
                return True
            else:
                logging.warning("No database connection or user ID for saving emotion data")
                return False
                
        except Exception as e:
            logging.error(f"Error saving emotion to database: {e}")
            # Try to rollback if there's an error
            try:
                self.db.session.rollback()
            except:
                pass
            return False

    def start_detection(self, callback=None, db=None, user_id=None):
        """Start the emotion detection"""
        if self.emotion_model is None:
            logging.error("Emotion model not loaded. Cannot start detection.")
            return False
            
        if self.face_detector is None:
            logging.error("Face detector not loaded. Cannot start detection.")
            return False
        
        # Set database and user context
        if db and user_id:
            self.set_user_context(db, user_id)
            
        self.is_running = True
        self.callback_function = callback
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        logging.info("Emotion detection started")
        return True

    def stop_detection(self):
        """Stop the emotion detection"""
        self.is_running = False
        
        # Save any pending emotions before stopping
        self._save_pending_emotions()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Emotion detection stopped")

    def _save_pending_emotions(self):
        """Save any pending emotions to database"""
        if self.pending_saves and self.db and self.user_id:
            logging.info(f"Saving {len(self.pending_saves)} pending emotions to database")
            for emotion_data in self.pending_saves:
                self._save_emotion_to_db(emotion_data['emotion'], emotion_data['duration'])
            self.pending_saves.clear()

    def _detection_loop(self):
        """Main detection loop - EXACTLY like reference code"""
        # Initialize camera
        self.cap = self._initialize_camera()
        if self.cap is None:
            logging.error("Failed to initialize camera")
            self.is_running = False
            return

        # Track emotion durations for database saving
        emotion_start_time = time.time()
        current_emotion_duration = 0
        last_emotion = "Neutral"

        while self.is_running:
            try:
                # Read frame - EXACTLY like reference code
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to read frame from camera")
                    break
                
                # Resize frame - EXACTLY like reference code  
                frame = cv2.resize(frame, (850, 720))
                
                # Convert to grayscale - EXACTLY like reference code
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces - EXACTLY like reference code
                num_faces = self.face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                # Process each face - EXACTLY like reference code
                current_emotion = "Neutral"
                for (x, y, w, h) in num_faces:
                    # Draw rectangle - EXACTLY like reference code
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                    
                    # Extract ROI - EXACTLY like reference code
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    
                    # Preprocess - EXACTLY like reference code
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                    # Predict emotions - EXACTLY like reference code
                    emotion_prediction = self.emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    current_emotion = self.emotion_dict[maxindex]
                    
                    # Display emotion - EXACTLY like reference code
                    cv2.putText(frame, current_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
                    # Update emotion tracking
                    if current_emotion != self.current_emotion:
                        # Calculate duration of previous emotion
                        current_time = time.time()
                        previous_emotion_duration = current_time - emotion_start_time
                        
                        # Save previous emotion to database if it lasted more than 1 second
                        if previous_emotion_duration >= 1 and last_emotion != "Neutral":
                            if self.db and self.user_id:
                                # Try to save directly, if fails add to pending
                                if not self._save_emotion_to_db(last_emotion, int(previous_emotion_duration)):
                                    self.pending_saves.append({
                                        'emotion': last_emotion,
                                        'duration': int(previous_emotion_duration)
                                    })
                        
                        # Update current emotion
                        self.current_emotion = current_emotion
                        self.emotion_stats[current_emotion] = self.emotion_stats.get(current_emotion, 0) + 1
                        emotion_start_time = current_time
                        last_emotion = current_emotion
                        
                        # Call callback with emotion update
                        if self.callback_function:
                            self.callback_function({
                                'type': 'emotion_detected',
                                'emotion': current_emotion,
                                'stats': self.emotion_stats.copy()
                            })

                # Save snapshot every 5 seconds (for statistics)
                current_time = time.time()
                if current_time - self.last_save_time >= 5:
                    self.last_save_time = current_time
                    if self.callback_function and len(num_faces) > 0:
                        self.callback_function({
                            'type': 'emotion_snapshot',
                            'emotion': self.current_emotion,
                            'stats': self.emotion_stats.copy(),
                            'timestamp': datetime.now()
                        })

                # Display frame - EXACTLY like reference code
                cv2.imshow('Emotion Detection', frame)
                
                # Check for quit command - EXACTLY like reference code
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                logging.error(f"Error in detection loop: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
        
        # Save final emotion duration before closing
        if last_emotion != "Neutral":
            final_duration = time.time() - emotion_start_time
            if final_duration >= 1:
                if self.db and self.user_id:
                    self._save_emotion_to_db(last_emotion, int(final_duration))
        
        # Save any pending emotions
        self._save_pending_emotions()
        
        # Cleanup - EXACTLY like reference code
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        logging.info("Detection loop ended")

    def get_emotion_stats(self):
        """Get current emotion statistics"""
        total = sum(self.emotion_stats.values())
        percentages = {}
        if total > 0:
            percentages = {emotion: round((count / total) * 100, 1) 
                          for emotion, count in self.emotion_stats.items()}
        
        return {
            'current_emotion': self.current_emotion,
            'stats': self.emotion_stats,
            'percentages': percentages,
            'total_detections': total,
            'is_running': self.is_running,
            'model_loaded': self.emotion_model is not None,
            'face_detector_loaded': self.face_detector is not None,
            'user_id_set': self.user_id is not None,
            'db_set': self.db is not None
        }

    def reset_stats(self):
        """Reset emotion statistics"""
        self.emotion_stats = {emotion: 0 for emotion in self.emotion_dict.values()}
        self.emotion_history = []
        self.current_emotion = "Neutral"
        self.pending_saves.clear()
        logging.info("Emotion statistics reset")

# Global instance
emotion_detector = EmotionDetector()