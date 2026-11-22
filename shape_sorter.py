import pygame
import sys
import random
import time
import math
from gtts import gTTS
import pygame.mixer
from mutagen.mp3 import MP3
import os
import threading

class ShapeSorter:
    def __init__(self):
        self.shapes = ['circle', 'square', 'triangle', 'rectangle']
        self.colors = {
            'circle': (255, 0, 0),      # Red
            'square': (0, 255, 0),      # Green
            'triangle': (0, 0, 255),    # Blue
            'rectangle': (255, 255, 0)  # Yellow
        }
        
        self.current_level = 1
        self.score = 0
        self.lives = 3
        self.game_active = False
        self.current_shape = None
        self.target_bin = None
        self.shapes_collected = 0
        self.cursor_position = [320, 240]
        self.cursor_speed = 5
        self.auto_move = False
        self.callback_function = None
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
    def speak_text(self, text):
        """Text to speech function"""
        try:
            myobj = gTTS(text=text, lang='en-us', tld='com', slow=False)
            filename = f"static/voice_files/shape_voice_{int(time.time())}.mp3"
            myobj.save(filename)
            
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up file after playing
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f"Error in shape speech: {e}")
    
    def start_game(self, callback=None):
        """Start the shape sorting game"""
        self.callback_function = callback
        self.game_active = True
        self.score = 0
        self.lives = 3
        self.current_level = 1
        self.shapes_collected = 0
        self.generate_new_shape()
        
        if self.callback_function:
            self.callback_function({
                'type': 'game_started',
                'level': self.current_level
            })
        
        self.speak_text(f"Level {self.current_level}. Sort the shapes!")
    
    def generate_new_shape(self):
        """Generate a new random shape"""
        self.current_shape = random.choice(self.shapes)
        self.target_bin = random.choice(self.shapes)
        
        # Ensure target bin is different from current shape for higher levels
        if self.current_level > 2 and self.current_shape == self.target_bin:
            available_bins = [shape for shape in self.shapes if shape != self.current_shape]
            self.target_bin = random.choice(available_bins)
        
        # Auto-move cursor for children with motor challenges
        self.auto_move = self.current_level <= 2
        
        if self.callback_function:
            self.callback_function({
                'type': 'new_shape',
                'shape': self.current_shape,
                'target_bin': self.target_bin,
                'auto_move': self.auto_move
            })
        
        self.speak_text(f"Take the {self.current_shape} to the {self.target_bin} bin")
    
    def move_cursor_toward_target(self, target_pos):
        """Move cursor automatically toward target position"""
        dx = target_pos[0] - self.cursor_position[0]
        dy = target_pos[1] - self.cursor_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > self.cursor_speed:
            self.cursor_position[0] += (dx / distance) * self.cursor_speed
            self.cursor_position[1] += (dy / distance) * self.cursor_speed
    
    def check_collision(self, cursor_pos, target_pos, radius=30):
        """Check if cursor collides with target"""
        distance = math.sqrt((cursor_pos[0] - target_pos[0])**2 + 
                           (cursor_pos[1] - target_pos[1])**2)
        return distance < radius
    
    def sort_shape(self, selected_bin):
        """Sort the current shape into a bin"""
        if not self.game_active:
            return
        
        if selected_bin == self.target_bin:
            # Correct sorting
            self.score += 10 * self.current_level
            self.shapes_collected += 1
            self.speak_text(f"Correct! {self.current_shape} goes in {self.target_bin} bin")
            
            if self.callback_function:
                self.callback_function({
                    'type': 'correct_sort',
                    'shape': self.current_shape,
                    'score': self.score,
                    'shapes_collected': self.shapes_collected
                })
            
            # Level progression
            if self.shapes_collected >= 5:
                self.current_level += 1
                self.shapes_collected = 0
                self.speak_text(f"Great! Level up! Now level {self.current_level}")
                
                if self.callback_function:
                    self.callback_function({
                        'type': 'level_up',
                        'level': self.current_level
                    })
        else:
            # Incorrect sorting
            self.lives -= 1
            self.speak_text(f"Try again. {self.current_shape} should go in {self.target_bin} bin")
            
            if self.callback_function:
                self.callback_function({
                    'type': 'incorrect_sort',
                    'shape': self.current_shape,
                    'correct_bin': self.target_bin,
                    'lives': self.lives
                })
            
            if self.lives <= 0:
                self.game_over()
                return
        
        # Generate new shape
        self.generate_new_shape()
    
    def game_over(self):
        """End the game"""
        self.game_active = False
        self.speak_text(f"Game over! Your score is {self.score}")
        
        if self.callback_function:
            self.callback_function({
                'type': 'game_over',
                'final_score': self.score,
                'level_reached': self.current_level
            })
    
    def get_game_state(self):
        """Get current game state"""
        return {
            'game_active': self.game_active,
            'current_shape': self.current_shape,
            'target_bin': self.target_bin,
            'score': self.score,
            'lives': self.lives,
            'level': self.current_level,
            'shapes_collected': self.shapes_collected,
            'cursor_position': self.cursor_position,
            'auto_move': self.auto_move
        }

# Global instance
shape_sorter = ShapeSorter()