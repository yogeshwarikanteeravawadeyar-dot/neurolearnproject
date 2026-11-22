from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from wtforms import Form, StringField, PasswordField, validators
import cv2
import numpy as np
import time
import pygame
from gtts import gTTS
from mutagen.mp3 import MP3
import speech_recognition as sr
import sounddevice as sd
import wave
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime, date
import threading

# Import all detectors
from eye_blink_detector import eye_blink_detector
from gesture_detector import gesture_detector
from emotion_detector import emotion_detector
from shape_sorter import shape_sorter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'neurolearn-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///neurolearn.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    progress = db.relationship('Progress', backref='user', lazy=True)
    emotions = db.relationship('EmotionData', backref='user', lazy=True)
    games = db.relationship('GameData', backref='user', lazy=True)

class Progress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    module = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    completed_at = db.Column(db.DateTime, default=datetime.utcnow)

class EmotionData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    emotion = db.Column(db.String(20), nullable=False)
    duration = db.Column(db.Integer, nullable=False)  # Duration in seconds
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<EmotionData {self.emotion} for user {self.user_id}>'

class GameData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    game_type = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    level = db.Column(db.Integer, nullable=False)
    played_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Forms
class RegistrationForm(Form):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email Address', [validators.Length(min=6, max=35)])
    password = PasswordField('New Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')

class LoginForm(Form):
    username = StringField('Username', [validators.DataRequired()])
    password = PasswordField('Password', [validators.DataRequired()])

# Global status variables
eye_blink_status = {
    'is_active': False,
    'current_letter': '',
    'typed_word': '',
    'last_action': ''
}

gesture_status = {
    'is_active': False,
    'current_gesture': '',
    'current_word': '',
    'gesture_history': [],
    'last_action': None
}

emotion_status = {
    'is_active': False,
    'current_emotion': 'Neutral',
    'emotion_stats': {},
    'percentages': {},
    'last_action': None
}

shape_game_status = {
    'is_active': False,
    'game_state': {},
    'last_action': None
}

# Callback functions
def eye_blink_callback(data):
    global eye_blink_status
    eye_blink_status['last_action'] = data
    
    if data['type'] == 'letter_typed':
        eye_blink_status['current_letter'] = data['letter']
        eye_blink_status['typed_word'] = data['current_word']
    elif data['type'] == 'word_completed':
        save_progress('eye_blink', 100)

def gesture_callback(data):
    global gesture_status
    gesture_status['last_action'] = data
    
    if data['type'] == 'gesture_detected':
        gesture_status['current_gesture'] = data['gesture']
        gesture_status['current_word'] = data['current_word']
        gesture_status['gesture_history'].append(data['gesture'])
        if len(gesture_status['gesture_history']) > 10:
            gesture_status['gesture_history'].pop(0)
        
        if len(gesture_status['current_word']) % 5 == 0:
            save_progress('gesture', min(100, len(gesture_status['current_word']) * 10))

def emotion_callback(data):
    global emotion_status
    emotion_status['last_action'] = data
    
    if data['type'] == 'emotion_detected':
        emotion_status['current_emotion'] = data['emotion']
        emotion_status['emotion_stats'] = data['stats']
    
    elif data['type'] == 'emotion_snapshot':
        # Save emotion data to database every 5 seconds
        if current_user.is_authenticated:
            emotion_data = EmotionData(
                user_id=current_user.id,
                emotion=data['emotion'],
                duration=5
            )
            db.session.add(emotion_data)
            db.session.commit()

def shape_game_callback(data):
    global shape_game_status
    shape_game_status['last_action'] = data
    
    if data['type'] == 'game_over' and current_user.is_authenticated:
        # Save game results
        game_data = GameData(
            user_id=current_user.id,
            game_type='shape_sorter',
            score=data['final_score'],
            level=data['level_reached']
        )
        db.session.add(game_data)
        db.session.commit()
        save_progress('shape_game', min(100, data['final_score']))

# Utility Functions
def Play(text):
    """Text to Speech Function"""
    try:
        myobj = gTTS(text=text, lang='en-us', tld='com', slow=False)
        filename = f"static/voice_files/voice_{int(time.time())}.mp3"
        myobj.save(filename)
        
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        song = MP3(filename)
        time.sleep(song.info.length)
        pygame.mixer.quit()
        
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"Error in Play function: {e}")

def save_progress(module, score):
    """Save user progress to database"""
    if current_user.is_authenticated:
        progress = Progress(user_id=current_user.id, module=module, score=score)
        db.session.add(progress)
        db.session.commit()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        try:
            user = User(username=form.username.data, 
                       email=form.email.data, 
                       password=form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Username or email already exists!', 'error')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == form.password.data:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    progress_data = Progress.query.filter_by(user_id=current_user.id).all()
    
    modules = {
        'gesture': {'completed': False, 'score': 0, 'last_attempt': None},
        'speech': {'completed': False, 'score': 0, 'last_attempt': None},
        'eye_blink': {'completed': False, 'score': 0, 'last_attempt': None},
        'autism': {'completed': False, 'score': 0, 'last_attempt': None},
        'emotion': {'completed': False, 'score': 0, 'last_attempt': None},
        'shape_game': {'completed': False, 'score': 0, 'last_attempt': None},
        'tower_game': {'completed': False, 'score': 0, 'last_attempt': None}
    }
    
    for progress in progress_data:
        if progress.module in modules:
            modules[progress.module]['completed'] = True
            modules[progress.module]['score'] = progress.score
            modules[progress.module]['last_attempt'] = progress.completed_at
    
    # Get emotion statistics
    emotion_data = EmotionData.query.filter_by(user_id=current_user.id).all()
    emotion_summary = {}
    for data in emotion_data:
        if data.emotion in emotion_summary:
            emotion_summary[data.emotion] += data.duration
        else:
            emotion_summary[data.emotion] = data.duration
    
    # Get game statistics
    game_data = GameData.query.filter_by(user_id=current_user.id).all()
    game_stats = {}
    for game in game_data:
        if game.game_type not in game_stats:
            game_stats[game.game_type] = []
        game_stats[game.game_type].append({'score': game.score, 'level': game.level, 'date': game.played_at})
    
    return render_template('dashboard.html', 
                         modules=modules, 
                         user=current_user,
                         emotion_summary=emotion_summary,
                         game_stats=game_stats)

# Eye Blink Module Routes (from previous implementation)
@app.route('/eye_blink')
@login_required
def eye_blink():
    global eye_blink_status
    return render_template('eye_blink.html', eye_blink_status=eye_blink_status)

@app.route('/start_eye_blink')
@login_required
def start_eye_blink():
    global eye_blink_status
    try:
        if not eye_blink_detector.is_running:
            eye_blink_detector.start_detection(callback=eye_blink_callback)
            eye_blink_status['is_active'] = True
            flash('Eye blink detection started! Look at the camera.', 'success')
        else:
            flash('Eye blink detection is already running.', 'info')
    except Exception as e:
        flash(f'Error starting eye blink detection: {str(e)}', 'error')
    return redirect(url_for('eye_blink'))

@app.route('/stop_eye_blink')
@login_required
def stop_eye_blink():
    global eye_blink_status
    try:
        if eye_blink_detector.is_running:
            eye_blink_detector.stop_detection()
            eye_blink_status['is_active'] = False
            flash('Eye blink detection stopped.', 'info')
        else:
            flash('Eye blink detection is not running.', 'info')
    except Exception as e:
        flash(f'Error stopping eye blink detection: {str(e)}', 'error')
    return redirect(url_for('eye_blink'))

@app.route('/eye_blink_status')
@login_required
def get_eye_blink_status():
    global eye_blink_status
    if eye_blink_detector.is_running:
        status = eye_blink_detector.get_status()
        eye_blink_status.update(status)
    return jsonify(eye_blink_status)
# Eye Blink Additional Routes
@app.route('/reset_eye_blink')
@login_required
def reset_eye_blink():
    global eye_blink_status
    try:
        eye_blink_detector.reset_typing()
        eye_blink_status['typed_word'] = ''
        eye_blink_status['current_letter'] = ''
        flash('Typing reset successfully.', 'success')
    except Exception as e:
        flash(f'Error resetting typing: {str(e)}', 'error')
    
    return redirect(url_for('eye_blink'))

@app.route('/speak_eye_blink_text')
@login_required
def speak_eye_blink_text():
    global eye_blink_status
    if eye_blink_status['typed_word']:
        Play(f"You typed: {eye_blink_status['typed_word']}")
        flash('Speaking the typed text.', 'info')
    else:
        flash('No text to speak.', 'warning')
    
    return redirect(url_for('eye_blink'))
# Gesture Module Routes (from previous implementation)
@app.route('/gesture')
@login_required
def gesture():
    global gesture_status
    return render_template('gesture.html', gesture_status=gesture_status)

@app.route('/start_gesture')
@login_required
def start_gesture():
    global gesture_status
    try:
        if not gesture_detector.is_running:
            gesture_detector.start_detection(callback=gesture_callback)
            gesture_status['is_active'] = True
            flash('Gesture recognition started! Show ASL letters to the camera.', 'success')
        else:
            flash('Gesture recognition is already running.', 'info')
    except Exception as e:
        flash(f'Error starting gesture recognition: {str(e)}', 'error')
    return redirect(url_for('gesture'))

@app.route('/stop_gesture')
@login_required
def stop_gesture():
    global gesture_status
    try:
        if gesture_detector.is_running:
            gesture_detector.stop_detection()
            gesture_status['is_active'] = False
            flash('Gesture recognition stopped.', 'info')
        else:
            flash('Gesture recognition is not running.', 'info')
    except Exception as e:
        flash(f'Error stopping gesture recognition: {str(e)}', 'error')
    return redirect(url_for('gesture'))

@app.route('/gesture_status')
@login_required
def get_gesture_status():
    global gesture_status
    if gesture_detector.is_running:
        status = gesture_detector.get_status()
        gesture_status.update(status)
    return jsonify(gesture_status)
# Gesture Learning Additional Routes
@app.route('/clear_gesture_word')
@login_required
def clear_gesture_word():
    global gesture_status
    try:
        gesture_detector.clear_word()
        gesture_status['current_word'] = ''
        gesture_status['current_gesture'] = ''
        flash('Word cleared successfully.', 'success')
    except Exception as e:
        flash(f'Error clearing word: {str(e)}', 'error')
    
    return redirect(url_for('gesture'))

@app.route('/speak_gesture_word')
@login_required
def speak_gesture_word():
    global gesture_status
    try:
        gesture_detector.speak_current_word()
        flash('Speaking the spelled word.', 'info')
    except Exception as e:
        flash(f'Error speaking word: {str(e)}', 'error')
    
    return redirect(url_for('gesture'))

@app.route('/speak_gesture_letter')
@login_required
def speak_gesture_letter():
    global gesture_status
    if gesture_status['current_gesture']:
        Play(f"Letter: {gesture_status['current_gesture']}")
        flash('Speaking the current letter.', 'info')
    else:
        flash('No letter detected yet.', 'warning')
    
    return redirect(url_for('gesture'))
## Emotion Detection Routes
@app.route('/emotion')
@login_required
def emotion():
    global emotion_status
    stats = emotion_detector.get_emotion_stats()
    emotion_status.update(stats)
    
    # Update user context for the emotion detector
    emotion_detector.set_user_context(db, current_user.id)
    
    return render_template('emotion.html', emotion_status=emotion_status)

@app.route('/start_emotion')
@login_required
def start_emotion():
    global emotion_status
    try:
        if not emotion_detector.is_running:
            # Pass database and user context to the emotion detector
            success = emotion_detector.start_detection(
                callback=emotion_callback, 
                db=db, 
                user_id=current_user.id
            )
            if success:
                emotion_status['is_active'] = True
                emotion_status['user_id_set'] = True
                emotion_status['db_set'] = True
                flash('Emotion detection started! Look at the camera.', 'success')
            else:
                flash('Failed to start emotion detection. Check if models are loaded.', 'error')
        else:
            flash('Emotion detection is already running.', 'info')
    except Exception as e:
        flash(f'Error starting emotion detection: {str(e)}', 'error')
    
    return redirect(url_for('emotion'))

@app.route('/stop_emotion')
@login_required
def stop_emotion():
    global emotion_status
    try:
        if emotion_detector.is_running:
            emotion_detector.stop_detection()
            emotion_status['is_active'] = False
            flash('Emotion detection stopped.', 'info')
        else:
            flash('Emotion detection is not running.', 'info')
    except Exception as e:
        flash(f'Error stopping emotion detection: {str(e)}', 'error')
    
    return redirect(url_for('emotion'))

@app.route('/emotion_status')
@login_required
def get_emotion_status():
    global emotion_status
    stats = emotion_detector.get_emotion_stats()
    emotion_status.update(stats)
    return jsonify(emotion_status)

@app.route('/reset_emotion_stats')
@login_required
def reset_emotion_stats():
    global emotion_status
    try:
        emotion_detector.reset_stats()
        emotion_status['emotion_stats'] = {}
        emotion_status['percentages'] = {}
        emotion_status['current_emotion'] = 'Neutral'
        flash('Emotion statistics reset.', 'success')
    except Exception as e:
        flash(f'Error resetting emotion stats: {str(e)}', 'error')
    
    return redirect(url_for('emotion'))

@app.route('/check_emotion_models')
@login_required
def check_emotion_models():
    """Check if emotion models are properly loaded"""
    stats = emotion_detector.get_emotion_stats()
    return jsonify({
        'model_loaded': stats['model_loaded'],
        'face_detector_loaded': stats['face_detector_loaded'],
        'is_running': stats['is_running'],
        'user_id_set': stats['user_id_set'],
        'db_set': stats['db_set']
    })

# Update the emotion callback function
def emotion_callback(data):
    """Callback function for emotion detector - now handles database saving"""
    global emotion_status
    emotion_status['last_action'] = data
    
    if data['type'] == 'emotion_detected':
        emotion_status['current_emotion'] = data['emotion']
        emotion_status['stats'] = data['stats']
        
        # Database saving is now handled within the emotion detector thread
        # using the pre-configured user_id and db instance
        
    elif data['type'] == 'emotion_snapshot':
        # This is just for statistics, database saving happens in real-time
        # in the detection loop based on emotion durations
        pass
# Shape Sorting Game Routes
@app.route('/shape_game')
@login_required
def shape_game():
    global shape_game_status
    shape_game_status['game_state'] = shape_sorter.get_game_state()
    return render_template('shape_game.html', shape_game_status=shape_game_status)

@app.route('/start_shape_game')
@login_required
def start_shape_game():
    global shape_game_status
    try:
        shape_sorter.start_game(callback=shape_game_callback)
        shape_game_status['is_active'] = True
        shape_game_status['game_state'] = shape_sorter.get_game_state()
        flash('Shape sorting game started!', 'success')
    except Exception as e:
        flash(f'Error starting shape game: {str(e)}', 'error')
    return redirect(url_for('shape_game'))

@app.route('/shape_game_status')
@login_required
def get_shape_game_status():
    global shape_game_status
    shape_game_status['game_state'] = shape_sorter.get_game_state()
    return jsonify(shape_game_status)

@app.route('/sort_shape/<bin_name>')
@login_required
def sort_shape(bin_name):
    global shape_game_status
    try:
        shape_sorter.sort_shape(bin_name)
        shape_game_status['game_state'] = shape_sorter.get_game_state()
        shape_game_status['last_action'] = {'type': 'shape_sorted', 'bin': bin_name}
        return jsonify({'success': True, 'game_state': shape_game_status['game_state']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Tower of Hanoi Game Route
@app.route('/tower_game')
@login_required
def tower_game():
    return render_template('tower_game.html')

@app.route('/save_tower_score', methods=['POST'])
@login_required
def save_tower_score():
    try:
        data = request.json
        game_data = GameData(
            user_id=current_user.id,
            game_type='tower_hanoi',
            score=data['score'],
            level=data['level']
        )
        db.session.add(game_data)
        db.session.commit()
        save_progress('tower_game', min(100, data['score']))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Speech Recognition Module (from previous implementation)
@app.route('/speech')
@login_required
def speech():
    return render_template('speech.html')

@app.route('/textspeech', methods=['POST'])
@login_required
def textspeech():
    try:
        check = request.json.get("text", "").lower()
        check = check.replace('.', '')
        
        Play(check)
        
        duration = 5
        fs = 44100
        channels = 2
        filename = "static/temp/input_audio.wav"
        
        os.makedirs("static/temp", exist_ok=True)
        
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype=np.int16)
        sd.wait()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio_data.tobytes())

        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data).lower()
                
                if os.path.exists(filename):
                    os.remove(filename)
                
                if text == check:
                    Play("Very good! You said it correctly. Next sentence is...")
                    save_progress('speech', 100)
                    return jsonify({"success": True, "message": "Correct!"})
                else:
                    Play("Sorry, that was incorrect. Please try again.")
                    save_progress('speech', 50)
                    return jsonify({"success": False, "message": "Incorrect, try again."})
                    
            except sr.UnknownValueError:
                Play("I couldn't understand what you said. Please try again.")
                return jsonify({"success": False, "message": "Speech not understood."})
            except sr.RequestError as e:
                Play("There was an error with the speech service.")
                return jsonify({"success": False, "message": f"Service error: {e}"})
                
    except Exception as e:
        return jsonify({"success": False, "message": "An error occurred."})

# Autism Prediction Module (from previous implementation)
@app.route('/autism')
@login_required
def autism():
    return render_template('autism.html')

@app.route('/predict_autism', methods=['POST'])
@login_required
def predict_autism():
    try:
        modela = joblib.load('models/best_model.pkl')
        data = request.form
        
        input_data = {
            "A1_Score": int(data.get('A1_Score', 0)),
            "A2_Score": int(data.get('A2_Score', 0)),
            # ... include all other fields from your autism form
            "Result": int(data.get('result', 0)),
        }
        
        input_data_df = pd.DataFrame([input_data])
        categorical_columns = ['Gender', 'Ethnicity', 'Jaundice', 'Autism', 'Country', 'Used_app_before', 'Age_desc', 'Relation']
        
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            input_data_df[col] = le.fit_transform([input_data[col]])[0]
        
        prediction = modela.predict(input_data_df)
        result = "Yes" if prediction[0] == 1 else "No"
        
        save_progress('autism', 100 if result == "No" else 50)
        
        if result == "Yes":
            recommendations = ["Consult with a pediatrician or autism specialist", ...]
        else:
            recommendations = ["Continue with regular developmental monitoring", ...]
        
        return render_template('autism_result.html', 
                             prediction=result, 
                             input_data=input_data,
                             recommendations=recommendations)
                             
    except Exception as e:
        flash('Error making prediction. Please try again.', 'error')
        return redirect(url_for('autism'))

@app.route('/shutdown')
def shutdown():
    """Shutdown the server and clean up resources"""
    if eye_blink_detector.is_running:
        eye_blink_detector.stop_detection()
    if gesture_detector.is_running:
        gesture_detector.stop_detection()
    if emotion_detector.is_running:
        emotion_detector.stop_detection()
    return 'Server shutting down...'

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    os.makedirs('static/voice_files', exist_ok=True)
    os.makedirs('static/temp', exist_ok=True)
    
    try:
        app.run(debug=True, use_reloader=False)
    finally:
        if eye_blink_detector.is_running:
            eye_blink_detector.stop_detection()
        if gesture_detector.is_running:
            gesture_detector.stop_detection()
        if emotion_detector.is_running:
            emotion_detector.stop_detection()
