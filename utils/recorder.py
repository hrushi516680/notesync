import cv2
import pyaudio
import wave
import threading
import speech_recognition as sr
import numpy as np
import os
from datetime import datetime
import time
import re
from collections import Counter
import logging
from contextlib import contextmanager
import queue
import json
from typing import Optional, Tuple, List
import platform
import subprocess

# NEW IMPORTS FOR KEYBOARD CONTROL
import keyboard
import signal
import sys

class VoiceActivatedLectureRecorder:
    def __init__(self):
        # Core state
        self.is_recording = False
        self.is_paused = False  # NEW: Pause functionality
        self.is_listening = True
        self.video_writer = None
        self.audio_frames = []
        
        # NEW: Control modes
        self.control_mode = "both"  # "voice", "keyboard", "both"
        self.push_to_talk_active = False
        self.keyboard_shortcuts_enabled = True
        
        # Recognition components
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Audio settings with fallback options
        self.audio_format = pyaudio.paInt16 
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.audio = None
        self.audio_stream = None
        
        # Threading components
        self.video_thread = None
        self.audio_thread = None
        self.command_thread = None
        self.keyboard_thread = None  # NEW: Keyboard listener thread
        self.stop_event = threading.Event()
        
        # File paths
        self.timestamp = None
        self.video_path = None
        self.audio_path = None
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = None
        self.pause_start_time = None
        self.total_pause_duration = 0
        
        # Command recognition improvement
        self.start_commands = ['start', 'begin', 'record', 'go']
        self.stop_commands = ['stop', 'end', 'finish', 'halt']
        self.pause_commands = ['pause', 'hold', 'wait']  # NEW
        self.resume_commands = ['resume', 'continue', 'unpause']  # NEW
        self.command_confidence_threshold = 0.7
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # NEW: Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = "recordings/logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = f"{log_dir}/recorder_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Enhanced Voice + Keyboard Recorder initialized")

    def _setup_keyboard_shortcuts(self):
        """NEW: Setup keyboard shortcuts and hotkeys"""
        try:
            # Global hotkeys (work even when app isn't focused)
            keyboard.add_hotkey('ctrl+shift+r', self._keyboard_toggle_recording)
            keyboard.add_hotkey('ctrl+shift+p', self._keyboard_toggle_pause)
            keyboard.add_hotkey('ctrl+shift+q', self._keyboard_quit)
            
            # Push-to-talk functionality
            keyboard.on_press_key('space', self._on_space_press)
            keyboard.on_release_key('space', self._on_space_release)
            
            # App-focused shortcuts (when terminal/app is active)
            keyboard.add_hotkey('r', self._keyboard_toggle_recording_simple)
            keyboard.add_hotkey('p', self._keyboard_toggle_pause_simple)
            keyboard.add_hotkey('q', self._keyboard_quit_simple)
            
            self.logger.info("⌨️ Keyboard shortcuts initialized")
            self._print_keyboard_shortcuts()
            
        except Exception as e:
            self.logger.error(f"❌ Keyboard shortcut setup failed: {e}")
            self.keyboard_shortcuts_enabled = False

    def _print_keyboard_shortcuts(self):
        """Print available keyboard shortcuts"""
        print("\n⌨️ KEYBOARD SHORTCUTS:")
        print("=" * 50)
        print("🌐 GLOBAL SHORTCUTS (work anywhere):")
        print("   Ctrl+Shift+R  → Start/Stop Recording")
        print("   Ctrl+Shift+P  → Pause/Resume Recording")
        print("   Ctrl+Shift+Q  → Quit Application")
        print("   Space (hold)  → Push-to-Talk Mode")
        print("\n🎯 APP SHORTCUTS (when terminal is active):")
        print("   R            → Start/Stop Recording")
        print("   P            → Pause/Resume Recording")
        print("   Q            → Quit Application")
        print("=" * 50)

    # NEW: Keyboard event handlers
    def _keyboard_toggle_recording(self):
        """Handle Ctrl+Shift+R - Toggle recording"""
        try:
            if not self.is_recording:
                print("⌨️🔴 Starting recording via keyboard...")
                self.logger.info("🎬 Recording started via keyboard shortcut")
                self.start_recording()
            else:
                print("⌨️⏹️ Stopping recording via keyboard...")
                self.logger.info("🛑 Recording stopped via keyboard shortcut")
                self.stop_recording()
        except Exception as e:
            self.logger.error(f"❌ Keyboard recording toggle failed: {e}")

    def _keyboard_toggle_pause(self):
        """Handle Ctrl+Shift+P - Toggle pause"""
        try:
            if self.is_recording:
                if not self.is_paused:
                    print("⌨️⏸️ Pausing recording via keyboard...")
                    self.logger.info("⏸️ Recording paused via keyboard shortcut")
                    self.pause_recording()
                else:
                    print("⌨️▶️ Resuming recording via keyboard...")
                    self.logger.info("▶️ Recording resumed via keyboard shortcut")
                    self.resume_recording()
            else:
                print("⚠️ No active recording to pause/resume")
        except Exception as e:
            self.logger.error(f"❌ Keyboard pause toggle failed: {e}")

    def _keyboard_quit(self):
        """Handle Ctrl+Shift+Q - Quit application"""
        print("⌨️👋 Quitting application via keyboard...")
        self.logger.info("🛑 Application quit via keyboard shortcut")
        self.cleanup()
        sys.exit(0)

    # Simple shortcuts (without modifiers)
    def _keyboard_toggle_recording_simple(self):
        """Handle R key - Simple recording toggle"""
        if not keyboard.is_pressed('ctrl') and not keyboard.is_pressed('shift'):
            self._keyboard_toggle_recording()

    def _keyboard_toggle_pause_simple(self):
        """Handle P key - Simple pause toggle"""
        if not keyboard.is_pressed('ctrl') and not keyboard.is_pressed('shift'):
            self._keyboard_toggle_pause()

    def _keyboard_quit_simple(self):
        """Handle Q key - Simple quit"""
        if not keyboard.is_pressed('ctrl') and not keyboard.is_pressed('shift'):
            self._keyboard_quit()

    # Push-to-talk functionality
    def _on_space_press(self, event):
        """Handle space key press - Start push-to-talk"""
        if not self.push_to_talk_active and not self.is_recording:
            self.push_to_talk_active = True
            print("🎤 Push-to-talk activated...")
            self.logger.info("🎤 Push-to-talk started")
            # Start temporary recording
            self._start_push_to_talk()

    def _on_space_release(self, event):
        """Handle space key release - End push-to-talk"""
        if self.push_to_talk_active:
            self.push_to_talk_active = False
            print("🎤 Push-to-talk deactivated")
            self.logger.info("🎤 Push-to-talk ended")
            # Process the push-to-talk audio for commands
            self._process_push_to_talk()

    def _start_push_to_talk(self):
        """Start temporary recording for push-to-talk"""
        try:
            self.push_to_talk_frames = []
            self.push_to_talk_recording = True
            
            # Start brief audio capture for command recognition
            threading.Thread(target=self._capture_push_to_talk_audio, daemon=True).start()
        except Exception as e:
            self.logger.error(f"❌ Push-to-talk start failed: {e}")

    def _capture_push_to_talk_audio(self):
        """Capture audio during push-to-talk"""
        try:
            with self._audio_stream_context() as stream:
                while self.push_to_talk_active:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    if hasattr(self, 'push_to_talk_frames'):
                        self.push_to_talk_frames.append(data)
                    time.sleep(0.01)
        except Exception as e:
            self.logger.error(f"❌ Push-to-talk audio capture failed: {e}")

    def _process_push_to_talk(self):
        """Process audio captured during push-to-talk"""
        try:
            if not hasattr(self, 'push_to_talk_frames') or not self.push_to_talk_frames:
                return
                
            # Create temporary audio file
            temp_audio_path = "temp_push_to_talk.wav"
            with wave.open(temp_audio_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.push_to_talk_frames))
            
            # Recognize speech from push-to-talk audio
            with sr.AudioFile(temp_audio_path) as source:
                audio = self.recognizer.record(source)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"🎤 Push-to-talk command: {command}")
                
                # Process the command
                self._process_voice_command(command)
                
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
        except sr.UnknownValueError:
            print("🎤 Push-to-talk: Could not understand audio")
        except Exception as e:
            self.logger.error(f"❌ Push-to-talk processing failed: {e}")
        finally:
            if hasattr(self, 'push_to_talk_frames'):
                delattr(self, 'push_to_talk_frames')

    def _process_voice_command(self, command: str):
        """Process a recognized voice command"""
        try:
            if self._is_start_command(command) and not self.is_recording:
                print("🔊 Starting recording via voice...")
                self.start_recording()
            elif self._is_stop_command(command) and self.is_recording:
                print("🔊 Stopping recording via voice...")
                self.stop_recording()
            elif self._is_pause_command(command) and self.is_recording and not self.is_paused:
                print("🔊 Pausing recording via voice...")
                self.pause_recording()
            elif self._is_resume_command(command) and self.is_recording and self.is_paused:
                print("🔊 Resuming recording via voice...")
                self.resume_recording()
        except Exception as e:
            self.logger.error(f"❌ Voice command processing failed: {e}")

    # NEW: Pause/Resume functionality
    def pause_recording(self):
        """Pause the current recording"""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            self.pause_start_time = time.time()
            print("⏸️ Recording paused")
            self.logger.info("⏸️ Recording paused")

    def resume_recording(self):
        """Resume the paused recording"""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            if self.pause_start_time:
                pause_duration = time.time() - self.pause_start_time
                self.total_pause_duration += pause_duration
                self.pause_start_time = None
            print("▶️ Recording resumed")
            self.logger.info("▶️ Recording resumed")

    def _initialize_components(self):
        """Initialize all components with proper error handling"""
        try:
            # Create recordings directory
            if not os.path.exists("recordings"):
                os.makedirs("recordings")
                self.logger.info("📁 Created recordings directory")

            # Initialize audio system
            self._initialize_audio()
            
            # Initialize microphone with optimal settings
            self._initialize_microphone()
            
            # Test camera availability
            self._test_camera()
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            raise

    def _initialize_audio(self):
        """Initialize PyAudio with error handling and device selection"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Get optimal audio device
            device_info = self._get_best_audio_device()
            if device_info:
                self.logger.info(f"🎤 Using audio device: {device_info['name']}")
            
        except Exception as e:
            self.logger.error(f"❌ Audio initialization failed: {e}")
            raise

    def _get_best_audio_device(self):
        """Find the best available audio input device"""
        try:
            default_device = self.audio.get_default_input_device_info()
            self.logger.info(f"🎵 Default audio device: {default_device['name']}")
            return default_device
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get default audio device: {e}")
            return None

    def _initialize_microphone(self):
        """Initialize microphone with optimal settings"""
        try:
            self.microphone = sr.Microphone()
            
            # Adjust recognizer settings for better accuracy
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.dynamic_energy_adjustment_damping = 0.15
            self.recognizer.dynamic_energy_ratio = 1.5
            self.recognizer.pause_threshold = 0.8
            self.recognizer.operation_timeout = None
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 0.8
            
            self.logger.info("🎙️ Microphone initialized with optimized settings")
            
        except Exception as e:
            self.logger.error(f"❌ Microphone initialization failed: {e}")
            raise

    def _test_camera(self):
        """Test camera availability and get optimal settings"""
        try:
            test_cap = cv2.VideoCapture(0)
            if not test_cap.isOpened():
                raise Exception("Camera not accessible")
            
            # Get camera properties
            width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(test_cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"📹 Camera available: {width}x{height} @ {fps}fps")
            test_cap.release()
            
        except Exception as e:
            self.logger.error(f"❌ Camera test failed: {e}")
            raise

    @contextmanager
    def _audio_stream_context(self):
        """Context manager for audio stream with guaranteed cleanup"""
        stream = None
        try:
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=None
            )
            yield stream
        except Exception as e:
            self.logger.error(f"❌ Audio stream error: {e}")
            raise
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass

    def _normalize_command(self, command: str) -> str:
        """Normalize and clean voice command"""
        if not command:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', command.lower().strip())
        
        # Remove common filler words that might interfere
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically']
        words = normalized.split()
        filtered_words = [w for w in words if w not in filler_words]
        
        return ' '.join(filtered_words)

    def _is_start_command(self, command: str) -> bool:
        """Check if command is a start command with fuzzy matching"""
        normalized = self._normalize_command(command)
        return any(start_cmd in normalized for start_cmd in self.start_commands)

    def _is_stop_command(self, command: str) -> bool:
        """Check if command is a stop command with fuzzy matching"""
        normalized = self._normalize_command(command)
        return any(stop_cmd in normalized for stop_cmd in self.stop_commands)

    def _is_pause_command(self, command: str) -> bool:
        """NEW: Check if command is a pause command"""
        normalized = self._normalize_command(command)
        return any(pause_cmd in normalized for pause_cmd in self.pause_commands)

    def _is_resume_command(self, command: str) -> bool:
        """NEW: Check if command is a resume command"""
        normalized = self._normalize_command(command)
        return any(resume_cmd in normalized for resume_cmd in self.resume_commands)

    def listen_for_commands(self):
        """Enhanced command listening with better error handling"""
        if self.control_mode in ["voice", "both"]:
            print("🎤 Voice commands enabled:")
            print("   ▶️  Start: 'start', 'begin', 'record', 'go'")
            print("   ⏸️  Pause: 'pause', 'hold', 'wait'")
            print("   ▶️  Resume: 'resume', 'continue', 'unpause'")
            print("   ⏹️  Stop: 'stop', 'end', 'finish', 'halt'")
        
        # Calibrate microphone for ambient noise
        if self.control_mode in ["voice", "both"]:
            try:
                with self.microphone as source:
                    print("🔧 Calibrating microphone for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                    print("✅ Microphone calibrated")
            except Exception as e:
                self.logger.warning(f"⚠️ Microphone calibration failed: {e}")

        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_listening and not self.stop_event.is_set():
            try:
                # Only listen for voice commands if voice control is enabled
                if self.control_mode in ["voice", "both"]:
                    with self.microphone as source:
                        # Listen with timeout to allow for graceful shutdown
                        audio = self.recognizer.listen(
                            source, 
                            timeout=1, 
                            phrase_time_limit=5
                        )
                    
                    # Recognize speech with confidence scoring when possible
                    command = self.recognizer.recognize_google(audio).lower()
                    self.logger.info(f"🔊 Command heard: '{command}'")
                    print(f"🔊 Heard: {command}")

                    # Process commands using the centralized processor
                    self._process_voice_command(command)

                    # Reset error counter on successful recognition
                    consecutive_errors = 0
                else:
                    # If only keyboard mode, just wait
                    time.sleep(0.1)

            except sr.WaitTimeoutError:
                # This is expected, just continue listening
                continue
                
            except sr.UnknownValueError:
                # Speech was unintelligible, continue listening
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print("⚠️ Having trouble hearing commands. Check microphone.")
                    self.logger.warning("Multiple consecutive speech recognition failures")
                    consecutive_errors = 0
                continue
                
            except sr.RequestError as e:
                self.logger.error(f"❌ Speech recognition service error: {e}")
                print(f"❌ Speech service error: {e}")
                time.sleep(2)  # Wait before retrying
                continue
                
            except Exception as e:
                self.logger.error(f"❌ Unexpected error in command listening: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print("❌ Too many errors. Stopping command listener.")
                    break
                continue

    def start_recording(self):
        """Enhanced recording start with comprehensive error handling"""
        try:
            self.is_recording = True
            self.is_paused = False  # Reset pause state
            self.total_pause_duration = 0  # Reset pause duration
            self.stop_event.clear()
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.start_time = time.time()
            self.frame_count = 0

            self.logger.info(f"🎬 Starting recording session: {self.timestamp}")

            # Initialize camera with enhanced error handling
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Cannot access camera")

            # Set optimal camera properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 20

            # Create video writer with error checking
            self.video_path = f"recordings/lecture_{self.timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, fps, (frame_width, frame_height)
            )
            
            if not self.video_writer.isOpened():
                raise Exception("Failed to initialize video writer")

            # Setup audio recording
            self.audio_path = f"recordings/lecture_{self.timestamp}.wav"
            self.audio_frames = []

            # Start recording threads
            self.video_thread = threading.Thread(target=self.record_video, daemon=True)
            self.audio_thread = threading.Thread(target=self.record_audio, daemon=True)
            
            self.video_thread.start()
            self.audio_thread.start()

            self.logger.info("✅ Recording threads started successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to start recording: {e}")
            print(f"❌ Recording failed to start: {e}")
            self.is_recording = False
            self._cleanup_recording_resources()
            raise

    def record_video(self):
        """Enhanced video recording with performance monitoring and pause support"""
        try:
            while self.is_recording and not self.stop_event.is_set():
                if self.is_paused:
                    time.sleep(0.1)  # Wait during pause
                    continue
                    
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("⚠️ Failed to read frame from camera")
                    continue

                # Add enhanced recording indicator
                current_time = time.time() - self.start_time if self.start_time else 0
                # Subtract pause time for accurate recording duration
                actual_recording_time = current_time - self.total_pause_duration
                time_str = f"{int(actual_recording_time // 60):02d}:{int(actual_recording_time % 60):02d}"
                
                # Add recording status overlay
                if self.is_paused:
                    cv2.putText(frame, "⏸️ PAUSED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "🔴 RECORDING", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.putText(frame, f"Time: {time_str}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add control hints
                cv2.putText(frame, "Ctrl+Shift+P: Pause | Ctrl+Shift+R: Stop", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Write frame only if not paused
                if not self.is_paused:
                    self.video_writer.write(frame)
                    self.frame_count += 1
                
                # Display frame
                cv2.imshow('Recording', frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("📺 Video recording stopped by user (Q key)")
                    break

        except Exception as e:
            self.logger.error(f"❌ Video recording error: {e}")
        finally:
            self._cleanup_video_resources()

    def record_audio(self):
        """Enhanced audio recording with better error handling and pause support"""
        try:
            with self._audio_stream_context() as stream:
                self.audio_stream = stream
                
                while self.is_recording and not self.stop_event.is_set():
                    try:
                        if self.is_paused:
                            time.sleep(0.1)  # Wait during pause
                            continue
                            
                        data = stream.read(self.chunk, exception_on_overflow=False)
                        self.audio_frames.append(data)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Audio chunk read error: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"❌ Audio recording error: {e}")

    def stop_recording(self):
        """Enhanced recording stop with comprehensive cleanup"""
        try:
            self.logger.info("🛑 Stopping recording...")
            self.is_recording = False
            self.is_paused = False
            self.stop_event.set()

            # Wait for threads to complete with timeout
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=5)
                if self.video_thread.is_alive():
                    self.logger.warning("⚠️ Video thread did not stop gracefully")

            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=5)
                if self.audio_thread.is_alive():
                    self.logger.warning("⚠️ Audio thread did not stop gracefully")

            # Save audio file with error handling
            self._save_audio_file()

            # Calculate recording statistics
            total_duration = time.time() - self.start_time if self.start_time else 0
            actual_recording_duration = total_duration - self.total_pause_duration
            avg_fps = self.frame_count / actual_recording_duration if actual_recording_duration > 0 else 0

            print("✅ Recording saved successfully!")
            print(f"📹 Video: {self.video_path}")
            print(f"🎵 Audio: {self.audio_path}")
            print(f"⏱️ Total Duration: {total_duration:.1f}s | Recording Duration: {actual_recording_duration:.1f}s")
            print(f"⏸️ Pause Time: {self.total_pause_duration:.1f}s | Frames: {self.frame_count} | Avg FPS: {avg_fps:.1f}")

            self.logger.info(f"📊 Recording stats - Total: {total_duration:.1f}s, Recording: {actual_recording_duration:.1f}s, Pause: {self.total_pause_duration:.1f}s, Frames: {self.frame_count}, Avg FPS: {avg_fps:.1f}")

            # Process recording (transcription, keywords, highlights)
            transcript = self.transcribe_audio()
            keywords = self.extract_keywords(transcript)
            highlights = self.find_highlights(transcript)

            # Save metadata
            self._save_metadata(transcript, keywords, highlights, actual_recording_duration)

            # Display results
            print("\n📝 Transcript:")
            print(transcript)
            print(f"\n🔑 Keywords: {', '.join(keywords)}")
            print(f"⭐ Highlights: {highlights}")

            return self.video_path, transcript, keywords, highlights

        except Exception as e:
            self.logger.error(f"❌ Error stopping recording: {e}")
            raise
        finally:
            self._cleanup_recording_resources()

    def _save_audio_file(self):
        """Save audio file with enhanced error handling"""
        try:
            if not self.audio_frames:
                self.logger.warning("⚠️ No audio frames to save")
                return

            with wave.open(self.audio_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.audio_frames))
            
            self.logger.info(f"🎵 Audio saved: {self.audio_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save audio: {e}")
            raise

    def _save_metadata(self, transcript: str, keywords: list, highlights: list, duration: float):
        """Save recording metadata to JSON file"""
        try:
            metadata = {
                'timestamp': self.timestamp,
                'video_path': self.video_path,
                'audio_path': self.audio_path,
                'duration': duration,
                'total_pause_duration': self.total_pause_duration,
                'frame_count': self.frame_count,
                'transcript': transcript,
                'keywords': keywords,
                'highlights': highlights,
                'recording_date': datetime.now().isoformat()
            }
            
            metadata_path = f"recordings/metadata_{self.timestamp}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📄 Metadata saved: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save metadata: {e}")

    def _cleanup_video_resources(self):
        """Clean up video recording resources"""
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            if hasattr(self, 'video_writer') and self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            self.logger.info("📹 Video resources cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Video cleanup error: {e}")

    def _cleanup_recording_resources(self):
        """Clean up all recording resources"""
        try:
            self._cleanup_video_resources()
            
            if hasattr(self, 'audio_stream') and self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except:
                    pass
            
            self.logger.info("🧹 All recording resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Resource cleanup error: {e}")

    def transcribe_audio(self) -> str:
        """Transcribe the recorded audio with enhanced error handling"""
        try:
            if not os.path.exists(self.audio_path):
                self.logger.warning("⚠️ Audio file not found for transcription")
                return "Transcription unavailable - audio file not found"

            print("🔄 Transcribing audio...")
            
            with sr.AudioFile(self.audio_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                audio = self.recognizer.record(source)
            
            # Use Google Speech Recognition
            transcript = self.recognizer.recognize_google(audio)
            self.logger.info("📝 Audio transcription completed")
            return transcript
            
        except sr.UnknownValueError:
            self.logger.warning("⚠️ Could not understand audio for transcription")
            return "Transcription unavailable - audio not clear enough"
        except sr.RequestError as e:
            self.logger.error(f"❌ Transcription service error: {e}")
            return f"Transcription error: {e}"
        except Exception as e:
            self.logger.error(f"❌ Transcription failed: {e}")
            return f"Transcription failed: {e}"

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from transcript using simple frequency analysis"""
        try:
            if not text or text.startswith("Transcription"):
                return []
            
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            
            # Remove common stop words
            stop_words = {
                'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know',
                'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
                'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over',
                'such', 'take', 'than', 'them', 'well', 'were', 'what', 'your'
            }
            
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count word frequency
            word_count = Counter(filtered_words)
            
            # Get most common words
            keywords = [word for word, count in word_count.most_common(max_keywords)]
            
            self.logger.info(f"🔑 Extracted {len(keywords)} keywords")
            return keywords
            
        except Exception as e:
            self.logger.error(f"❌ Keyword extraction failed: {e}")
            return []

    def find_highlights(self, text: str, max_highlights: int = 3) -> List[str]:
        """Find potential highlights in the transcript"""
        try:
            if not text or text.startswith("Transcription"):
                return []
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not sentences:
                return []
            
            # Simple highlight detection based on sentence length and keywords
            highlight_indicators = [
                'important', 'key', 'remember', 'note', 'crucial', 'significant',
                'main', 'primary', 'essential', 'fundamental', 'critical'
            ]
            
            scored_sentences = []
            for sentence in sentences:
                score = len(sentence)  # Longer sentences might be more detailed
                
                # Boost score for highlight indicators
                for indicator in highlight_indicators:
                    if indicator in sentence.lower():
                        score += 100
                
                scored_sentences.append((sentence, score))
            
            # Sort by score and take top highlights
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            highlights = [sentence for sentence, score in scored_sentences[:max_highlights]]
            
            self.logger.info(f"⭐ Found {len(highlights)} highlights")
            return highlights
            
        except Exception as e:
            self.logger.error(f"❌ Highlight extraction failed: {e}")
            return []

    def cleanup(self):
        """Comprehensive cleanup of all resources"""
        try:
            self.logger.info("🧹 Starting comprehensive cleanup...")
            
            # Stop all operations
            self.is_listening = False
            self.is_recording = False
            self.stop_event.set()
            
            # Clean up recording resources
            self._cleanup_recording_resources()
            
            # Clean up PyAudio
            if hasattr(self, 'audio') and self.audio:
                self.audio.terminate()
            
            # Remove keyboard hooks
            if self.keyboard_shortcuts_enabled:
                try:
                    keyboard.unhook_all()
                    self.logger.info("⌨️ Keyboard hooks removed")
                except:
                    pass
            
            self.logger.info("✅ Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")

    def run(self):
        """Main application loop with enhanced control flow"""
        try:
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            print("🚀 Enhanced Voice + Keyboard Activated Lecture Recorder")
            print("=" * 60)
            print(f"🎛️ Control Mode: {self.control_mode}")
            
            if self.keyboard_shortcuts_enabled:
                self._print_keyboard_shortcuts()
            
            print("\n🎤 Voice Commands:")
            print("   ▶️  Start: 'start', 'begin', 'record', 'go'")
            print("   ⏸️  Pause: 'pause', 'hold', 'wait'")
            print("   ▶️  Resume: 'resume', 'continue', 'unpause'")
            print("   ⏹️  Stop: 'stop', 'end', 'finish', 'halt'")
            print("\n🎯 Ready! Say a command or use keyboard shortcuts...")
            
            # Start command listening thread
            if self.control_mode in ["voice", "both"]:
                self.command_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
                self.command_thread.start()
            
            # Main loop
            while self.is_listening and not self.stop_event.is_set():
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    break
            
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
        except Exception as e:
            self.logger.error(f"❌ Application error: {e}")
            print(f"❌ Application error: {e}")
        finally:
            self.cleanup()

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.logger.info(f"🛑 Received signal {signum}")
        self.is_listening = False
        self.stop_event.set()
        if self.is_recording:
            self.stop_recording()


def main():
    """Main entry point with configuration options"""
    try:
        print("🎬 Initializing Enhanced Voice + board Lecture Recorder...")
        
        # Create and configure recorder
        recorder = VoiceActivatedLectureRecorder()
        
        # You can customize control mode here:
        # recorder.control_mode = "voice"     # Voice only
        # recorder.control_mode = "keyboard"  # Keyboard only
        # recorder.control_mode = "both"      # Both (default)
        
        # Run the application
        recorder.run()
        
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        logging.error(f"❌ Application startup failed: {e}")
    finally:
        print("🏁 Application terminated")


if __name__ == "__main__":
    main()