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

class VoiceActivatedLectureRecorder:
    def __init__(self):
        # Core state
        self.is_recording = False
        self.is_listening = True
        self.video_writer = None
        self.audio_frames = []
        
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
        self.stop_event = threading.Event()
        
        # File paths
        self.timestamp = None
        self.video_path = None
        self.audio_path = None
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = None
        
        # Command recognition improvement
        self.start_commands = ['start', 'begin', 'record', 'go']
        self.stop_commands = ['stop', 'end', 'finish', 'halt']
        self.command_confidence_threshold = 0.7
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()

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
        self.logger.info("🚀 Voice-Activated Lecture Recorder initialized")

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

    def listen_for_commands(self):
        """Enhanced command listening with better error handling"""
        print("🎤 Say 'start' to begin recording and 'stop' to end.")
        print("💡 Commands: start/begin/record/go | stop/end/finish/halt")
        
        # Calibrate microphone for ambient noise
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

                # Process commands
                if self._is_start_command(command) and not self.is_recording:
                    print("▶️ Starting recording...")
                    self.logger.info("🎬 Recording started via voice command")
                    self.start_recording()
                    
                elif self._is_stop_command(command) and self.is_recording:
                    print("⏹️ Stopping recording...")
                    self.logger.info("🛑 Recording stopped via voice command")
                    return self.stop_recording()

                # Reset error counter on successful recognition
                consecutive_errors = 0

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
        """Enhanced video recording with performance monitoring"""
        try:
            while self.is_recording and not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("⚠️ Failed to read frame from camera")
                    continue

                # Add enhanced recording indicator
                current_time = time.time() - self.start_time if self.start_time else 0
                time_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
                
                # Add recording status overlay
                cv2.putText(frame, "🔴 RECORDING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Time: {time_str}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame
                self.video_writer.write(frame)
                self.frame_count += 1
                
                # Display frame
                cv2.imshow('Recording', frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("📺 Video recording stopped by user (Q key)")
                    break

        except Exception as e:
            self.logger.error(f"❌ Video recording error: {e}")
        finally:
            self._cleanup_video_resources()

    def record_audio(self):
        """Enhanced audio recording with better error handling"""
        try:
            with self._audio_stream_context() as stream:
                self.audio_stream = stream
                
                while self.is_recording and not self.stop_event.is_set():
                    try:
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
            duration = time.time() - self.start_time if self.start_time else 0
            avg_fps = self.frame_count / duration if duration > 0 else 0

            print("✅ Recording saved successfully!")
            print(f"📹 Video: {self.video_path}")
            print(f"🎵 Audio: {self.audio_path}")
            print(f"⏱️ Duration: {duration:.1f}s | Frames: {self.frame_count} | Avg FPS: {avg_fps:.1f}")

            self.logger.info(f"📊 Recording stats - Duration: {duration:.1f}s, Frames: {self.frame_count}, Avg FPS: {avg_fps:.1f}")

            # Process recording (transcription, keywords, highlights)
            transcript = self.transcribe_audio()
            keywords = self.extract_keywords(transcript)
            highlights = self.find_highlight(transcript)

            # Save metadata
            self._save_metadata(transcript, keywords, highlights, duration)

            # Display results
            print("\n📝 Transcript:")
            print(transcript)
            print(f"\n🔑 Keywords: {', '.join(keywords)}")
            print(f"⭐ Highlight: {highlights}")

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

            # Verify file was created and has content
            if os.path.exists(self.audio_path) and os.path.getsize(self.audio_path) > 0:
                self.logger.info(f"💾 Audio saved: {self.audio_path}")
            else:
                self.logger.error("❌ Audio file was not saved properly")

        except Exception as e:
            self.logger.error(f"❌ Failed to save audio file: {e}")

    def _save_metadata(self, transcript: str, keywords: List[str], highlights: str, duration: float):
        """Save recording metadata to JSON file"""
        try:
            metadata = {
                'timestamp': self.timestamp,
                'duration': duration,
                'frame_count': self.frame_count,
                'video_path': self.video_path,
                'audio_path': self.audio_path,
                'transcript': transcript,
                'keywords': keywords,
                'highlights': highlights,
                'created_at': datetime.now().isoformat()
            }

            metadata_path = f"recordings/lecture_{self.timestamp}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.logger.info(f"📄 Metadata saved: {metadata_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save metadata: {e}")

    def _cleanup_video_resources(self):
        """Cleanup video-related resources"""
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            if hasattr(self, 'video_writer') and self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up video resources: {e}")

    def _cleanup_recording_resources(self):
        """Cleanup all recording-related resources"""
        try:
            self._cleanup_video_resources()
            
            if hasattr(self, 'audio_stream') and self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up recording resources: {e}")

    def transcribe_audio(self):
        """Enhanced audio transcription with better error handling"""
        print("🧠 Transcribing audio...")
        self.logger.info("🔤 Starting audio transcription")
        
        try:
            if not os.path.exists(self.audio_path):
                error_msg = "Audio file not found for transcription"
                self.logger.error(f"❌ {error_msg}")
                return f"❌ {error_msg}"

            # Check file size
            file_size = os.path.getsize(self.audio_path)
            if file_size == 0:
                error_msg = "Audio file is empty"
                self.logger.error(f"❌ {error_msg}")
                return f"❌ {error_msg}"

            self.logger.info(f"📊 Audio file size: {file_size / 1024 / 1024:.2f} MB")

            # Transcribe with enhanced settings
            with sr.AudioFile(self.audio_path) as source:
                # Adjust for noise if needed
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.record(source)

            # Attempt transcription with Google Speech Recognition
            text = self.recognizer.recognize_google(
                audio, 
                language='en-US',
                show_all=False
            )

            self.logger.info("✅ Transcription completed successfully")
            return text

        except sr.UnknownValueError:
            error_msg = "Could not understand audio content"
            self.logger.warning(f"⚠️ {error_msg}")
            return f"⚠️ {error_msg}"
            
        except sr.RequestError as e:
            error_msg = f"Speech recognition service error: {e}"
            self.logger.error(f"❌ {error_msg}")
            return f"❌ {error_msg}"
            
        except FileNotFoundError:
            error_msg = "Audio file not found"
            self.logger.error(f"❌ {error_msg}")
            return f"❌ {error_msg}"
            
        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            return f"❌ {error_msg}"

    def extract_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction with better filtering"""
        if not text or text.startswith('❌') or text.startswith('⚠️'):
            return []

        try:
            # Clean and tokenize text
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Enhanced stop words list
            stop_words = {
                'the', 'and', 'to', 'of', 'a', 'in', 'is', 'for', 'on', 'that', 'with', 
                'as', 'are', 'was', 'will', 'be', 'by', 'at', 'from', 'up', 'about', 
                'into', 'through', 'during', 'before', 'after', 'above', 'below', 'out', 
                'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
                'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 
                'so', 'than', 'too', 'very', 'can', 'could', 'should', 'would', 'have', 
                'has', 'had', 'do', 'does', 'did', 'get', 'got', 'make', 'made', 'take', 
                'took', 'come', 'came', 'go', 'went', 'see', 'saw', 'know', 'knew', 
                'think', 'thought', 'say', 'said', 'tell', 'told', 'give', 'gave', 
                'find', 'found', 'work', 'worked', 'call', 'called', 'try', 'tried',
                'like', 'just', 'now', 'well', 'also', 'back', 'still', 'way', 'even',
                'need', 'really', 'thing', 'things', 'people', 'time', 'times', 'year',
                'years', 'day', 'days', 'good', 'new', 'first', 'last', 'long', 'great',
                'little', 'right', 'old', 'different', 'small', 'large', 'next', 'early',
                'young', 'important', 'few', 'public', 'bad', 'same', 'able'
            }
            
            # Filter words
            keywords = [w for w in words if w not in stop_words and len(w) >= 4]
            
            # Count frequency and get top keywords
            freq = Counter(keywords)
            top_keywords = [kw for kw, count in freq.most_common(8) if count >= 1]
            
            self.logger.info(f"🔑 Extracted {len(top_keywords)} keywords from {len(words)} words")
            
            return top_keywords[:5]  # Return top 5 as originally intended

        except Exception as e:
            self.logger.error(f"❌ Keyword extraction failed: {e}")
            return []

    def find_highlight(self, text: str) -> str:
        """Enhanced highlight detection with multiple criteria"""
        if not text or text.startswith('❌') or text.startswith('⚠️'):
            return "No transcript available for highlight detection."

        try:
            # Split into sentences more intelligently
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return "No sentences found for highlight detection."

            # Enhanced highlight keywords with weights
            highlight_patterns = {
                'importance': ['important', 'crucial', 'critical', 'key', 'essential', 'vital', 'significant'],
                'exam': ['exam', 'test', 'quiz', 'assessment', 'evaluation', 'grade'],
                'attention': ['remember', 'note', 'notice', 'pay attention', 'focus', 'highlight'],
                'emphasis': ['especially', 'particularly', 'specifically', 'notably', 'mainly'],
                'conclusion': ['conclusion', 'summary', 'in summary', 'to conclude', 'therefore'],
                'definition': ['definition', 'means', 'defined as', 'refers to', 'is when']
            }

            # Score sentences based on highlight patterns
            scored_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = 0
                
                for category, patterns in highlight_patterns.items():
                    for pattern in patterns:
                        if pattern in sentence_lower:
                            # Weight different categories
                            if category in ['importance', 'exam']:
                                score += 3
                            elif category in ['attention', 'emphasis']:
                                score += 2
                            else:
                                score += 1
                
                if score > 0:
                    scored_sentences.append((sentence.strip(), score))

            # Return highest scoring sentence
            if scored_sentences:
                best_sentence = max(scored_sentences, key=lambda x: x[1])
                self.logger.info(f"⭐ Found highlight with score {best_sentence[1]}")
                return best_sentence[0]
            else:
                # Fallback: return first sentence if it's substantial
                if sentences and len(sentences[0]) > 20:
                    return sentences[0]
                return "No specific highlights detected in the recording."

        except Exception as e:
            self.logger.error(f"❌ Highlight detection failed: {e}")
            return "Error occurred during highlight detection."

    def cleanup(self):
        """Enhanced cleanup with comprehensive resource management"""
        try:
            self.logger.info("🧹 Starting cleanup process")
            
            # Stop listening
            self.is_listening = False
            self.stop_event.set()
            
            # Stop recording if active
            if self.is_recording:
                self.is_recording = False
                time.sleep(0.5)  # Give threads time to notice
            
            # Clean up recording resources
            self._cleanup_recording_resources()
            
            # Terminate audio system
            if self.audio:
                try:
                    self.audio.terminate()
                    self.logger.info("🔊 Audio system terminated")
                except Exception as e:
                    self.logger.error(f"❌ Error terminating audio: {e}")
            
            self.logger.info("✅ Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")

    def get_system_info(self):
        """Get system information for debugging"""
        try:
            info = {
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'opencv_version': cv2.__version__,
            }
            
            # Audio device info
            if self.audio:
                info['audio_devices'] = []
                for i in range(self.audio.get_device_count()):
                    try:
                        device_info = self.audio.get_device_info_by_index(i)
                        if device_info['maxInputChannels'] > 0:
                            info['audio_devices'].append({
                                'name': device_info['name'],
                                'index': i,
                                'channels': device_info['maxInputChannels']
                            })
                    except:
                        continue
            
            self.logger.info(f"💻 System info: {info}")
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system info: {e}")
            return {}

def main():
    """Enhanced main function with comprehensive error handling"""
    print("🎙️ Enhanced Voice Command Recorder with Transcription")
    print("=" * 60)
    
    recorder = None
    try:
        recorder = VoiceActivatedLectureRecorder()
        
        # Display system information
        print("\n💻 System Information:")
        system_info = recorder.get_system_info()
        if system_info:
            print(f"Platform: {system_info.get('platform', 'Unknown')}")
            print(f"Python: {system_info.get('python_version', 'Unknown')}")
            print(f"OpenCV: {system_info.get('opencv_version', 'Unknown')}")
            
            audio_devices = system_info.get('audio_devices', [])
            if audio_devices:
                print(f"Audio Input Devices: {len(audio_devices)} found")
            else:
                print("⚠️ No audio input devices detected")
        
        print("\n🚀 Starting voice command listener...")
        print("📋 Available Commands:")
        print("   ▶️  Start: 'start', 'begin', 'record', 'go'")
        print("   ⏹️  Stop: 'stop', 'end', 'finish', 'halt'")
        print("   🔤 Press 'Q' during recording to quit video")
        print("   ⌨️  Press Ctrl+C to exit application")
        print("-" * 60)
        
        # Start command listening
        recorder.listen_for_commands()
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
        if recorder:
            recorder.logger.info("🛑 Application interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if recorder:
            recorder.logger.error(f"💥 Fatal error: {e}")
        raise
    finally:
        if recorder:
            print("\n🧹 Cleaning up resources...")
            recorder.cleanup()
            print("👋 Goodbye!")

if __name__ == "__main__":
    main()