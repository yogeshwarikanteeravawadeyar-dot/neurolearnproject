import cv2
import numpy as np
import dlib
from math import hypot
import time
import threading
import json
import os
from flask import current_app
import pygame
from gtts import gTTS
from mutagen.mp3 import MP3

class EyeBlinkDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.board = np.zeros((300, 700), np.uint8)
        self.board[:] = 255
        
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Shape predictor file not found: {predictor_path}")
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Keyboard settings
        self.keyboard = np.zeros((1000, 800, 3), np.uint8)
        
        self.keys_set_1 = {
            0: "1", 1: "2", 2: "3", 3: "E",
            4: "4", 5: "5", 6: "6", 7: "C",
            8: "7", 9: "8", 10: "9", 11: "0"
        }
        
        # Counters
        self.frames = 0
        self.letter_index = 0
        self.blinking_frames = 0
        self.frames_to_blink = 6
        self.frames_active_letter = 9
        
        # Text and keyboard settings
        self.text = ""
        self.text1 = []
        self.keyboard_selected = "left"
        self.last_keyboard_selected = "left"
        self.select_keyboard_menu = True
        self.keyboard_selection_frames = 0
        self.count = 0
        
        # Communication with Flask
        self.current_letter = ""
        self.typed_word = ""
        self.is_running = False
        self.callback_function = None
        
        # Password for demonstration (you can make this configurable)
        self.password = ['1', '5']
        
    def midpoint(self, p1, p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
    
    def get_blinking_ratio(self, eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
        ratio = hor_line_length / ver_line_length
        return ratio
    
    def eyes_contour_points(self, facial_landmarks):
        left_eye = []
        right_eye = []
        for n in range(36, 42):
            x = facial_landmarks.part(n).x
            y = facial_landmarks.part(n).y
            left_eye.append([x, y])
        for n in range(42, 48):
            x = facial_landmarks.part(n).x
            y = facial_landmarks.part(n).y
            right_eye.append([x, y])
        left_eye = np.array(left_eye, np.int32)
        right_eye = np.array(right_eye, np.int32)
        return left_eye, right_eye
    
    def get_gaze_ratio(self, eye_points, facial_landmarks, frame, gray):
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
        
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
        
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        
        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)
        
        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white
        return gaze_ratio
    
    def draw_letters(self, letter_index, text, letter_light):
        # Keys positions
        positions = [
            (0, 0), (200, 0), (400, 0), (600, 0),
            (0, 200), (200, 200), (400, 200), (600, 200),
            (0, 400), (200, 400), (400, 400), (600, 400)
        ]
        
        if letter_index < len(positions):
            x, y = positions[letter_index]
            width = 200
            height = 200
            th = 3  # thickness
            
            # Text settings
            font_letter = cv2.FONT_HERSHEY_PLAIN
            font_scale = 9
            font_th = 4
            text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
            width_text, height_text = text_size[0], text_size[1]
            text_x = int((width - width_text) / 2) + x
            text_y = int((height + height_text) / 2) + y
            
            if letter_light:
                cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
                cv2.putText(self.keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
            else:
                cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
                cv2.putText(self.keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)
    
##    def speak_text(self, text):
##        """Text to speech function"""
##        try:
##            myobj = gTTS(text=text, lang='en-us', tld='com', slow=False)
##            filename = f"static/voice_files/voice_{int(time.time())}.mp3"
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
##            print(f"Error in speech: {e}")
    
    def start_detection(self, callback=None):
        """Start the eye blink detection"""
        self.is_running = True
        self.callback_function = callback
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop the eye blink detection"""
        self.is_running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    def reset_typing(self):
        """Reset the typed text"""
        self.typed_word = ""
        self.text1 = []
        self.count = 0
        if self.callback_function:
            self.callback_function({
                'type': 'typing_reset',
                'timestamp': time.time()
            })

    def speak_text(self, text):
        """Text to speech function"""
        try:
            myobj = gTTS(text=text, lang='en-us', tld='com', slow=False)
            filename = f"static/voice_files/eye_blink_voice_{int(time.time())}.mp3"
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
            print(f"Error in eye blink speech: {e}")    
    def _detection_loop(self):
        """Main detection loop"""
        font = cv2.FONT_HERSHEY_PLAIN
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                rows, cols, _ = frame.shape
                self.keyboard[:] = (26, 26, 26)
                self.frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Draw a white space for loading bar
                frame[rows - 50: rows, 0: cols] = (255, 255, 255)
                
                # Keyboard selected
                if self.keyboard_selected == "left":
                    keys_set = self.keys_set_1
                active_letter = keys_set[self.letter_index]
                self.current_letter = active_letter
                
                # Face detection
                faces = self.detector(gray)
                for face in faces:
                    landmarks = self.predictor(gray, face)
                    
                    left_eye, right_eye = self.eyes_contour_points(landmarks)
                    
                    # Detect blinking
                    left_eye_ratio = self.get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
                    right_eye_ratio = self.get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
                    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                    
                    # Eyes color
                    cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
                    cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)
                    
                    if self.select_keyboard_menu:
                        # Detecting gaze to select Left or Right keyboard
                        gaze_ratio_left_eye = self.get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, frame, gray)
                        gaze_ratio_right_eye = self.get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, frame, gray)
                        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
                        
                        if gaze_ratio <= 5:
                            self.keyboard_selected = "right"
                            self.keyboard_selection_frames += 1
                            if self.keyboard_selection_frames == 15:
                                self.select_keyboard_menu = False
                                self.frames = 0
                                self.keyboard_selection_frames = 0
                            if self.keyboard_selected != self.last_keyboard_selected:
                                self.last_keyboard_selected = self.keyboard_selected
                                self.keyboard_selection_frames = 0
                    else:
                        # Detect the blinking to select the key that is lighting up
                        if blinking_ratio > 5:
                            self.blinking_frames += 1
                            self.frames -= 1
                            
                            # Show green eyes when closed
                            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
                            
                            # Typing letter
                            if self.blinking_frames == self.frames_to_blink:
                                if active_letter != "E" and active_letter != "C":
                                    self.count += 1
                                    self.text += active_letter
                                    self.text1.append(active_letter)
                                    self.typed_word += active_letter
                                    
                                    # Speak the typed letter
                                    self.speak_text(active_letter)
                                    
                                    # Call callback function if provided
                                    if self.callback_function:
                                        self.callback_function({
                                            'type': 'letter_typed',
                                            'letter': active_letter,
                                            'current_word': self.typed_word
                                        })
                                    
                                    print(f'text1: {self.text1}')
                                    
                                # if active_letter == "E" or active_letter == "C":
                                #     if active_letter == "E":  # Enter key
                                #         if self.typed_word:
                                #             self.speak_text(f"You typed: {self.typed_word}")
                                #             if self.callback_function:
                                #                 self.callback_function({
                                #                     'type': 'word_completed',
                                #                     'word': self.typed_word
                                #                 })
                                #     else:  # Clear key
                                #         if self.typed_word:
                                #             self.typed_word = self.typed_word[:-1]
                                #             self.speak_text("Cleared")
                                    
                                    # self.count = max(0, self.count - 1)
                                    # if self.text1:
                                    #     self.text1.pop()
                                    # print(f'Deleted: {self.text1}')
                                
                                # # Password check (for demonstration)
                                # if len(self.text1) == 2:
                                #     if self.text1 == self.password:
                                #         self.speak_text("Password matched! Access granted.")
                                #         print('Password matched')
                                #         if self.callback_function:
                                #             self.callback_function({
                                #                 'type': 'password_matched',
                                #                 'password': ''.join(self.text1)
                                #             })
                                #     else:
                                #         self.speak_text("Password not matched. Try again.")
                                #         print('Password not matched')
                                #         if self.callback_function:
                                #             self.callback_function({
                                #                 'type': 'password_failed',
                                #                 'attempted': ''.join(self.text1)
                                #             })
                                    
                                #     self.text1 = []
                                #     self.count = 0
                                
                                # self.select_keyboard_menu = True
                        else:
                            self.blinking_frames = 0
                
                # Display letters on the keyboard
                if not self.select_keyboard_menu:
                    if self.frames == self.frames_active_letter:
                        self.letter_index += 1
                        self.frames = 0
                    if self.letter_index >= len(keys_set):
                        self.letter_index = 0
                    
                    for i in range(len(keys_set)):
                        light = (i == self.letter_index)
                        self.draw_letters(i, keys_set[i], light)
                
                # Show the text we're writing on the board
                self.board[:] = 255  # Clear board
                cv2.putText(self.board, self.typed_word, (80, 100), font, 3, 0, 3)
                
                # Blinking loading bar
                percentage_blinking = self.blinking_frames / self.frames_to_blink
                loading_x = int(cols * percentage_blinking)
                cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)
                
                # Display frames
                cv2.imshow("Eye Blink Detection", frame)
                cv2.imshow("Virtual Keyboard", self.keyboard)
                cv2.imshow("Typed Text", self.board)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Error in detection loop: {e}")
                continue
        
        # Cleanup
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    
    def get_status(self):
        """Get current status of the detector"""
        return {
            'is_running': self.is_running,
            'current_letter': self.current_letter,
            'typed_word': self.typed_word,
            'text1': self.text1
        }
    
    def reset_typing(self):
        """Reset the typed text"""
        self.typed_word = ""
        self.text1 = []
        self.count = 0

# Global instance
eye_blink_detector = EyeBlinkDetector()
