import numpy as np
import sys
import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
import ffmpeg
import audioop
from processing import *

class Recorder():
    def __init__(self,width,height,isLive,processing):
        self.processing = processing
        self.width = width
        self.height = height
        self.video_thread = self.VideoRecorder(self, "temp", (width, height),isLive=isLive)
        self.audio_thread = self.AudioRecorder(self, "temp",isLive=isLive)


    def changeIsLive(self,newVal):
        self.video_thread.isLive = newVal
        self.audio_thread.isLive = newVal

    def recording(self):
        return self.video_thread.open or self.audio_thread.open

    def startRecording(self):
        self.video_thread.start()
        self.audio_thread.start()

    def stopRecording(self):
        self.video_thread.stop()
        self.audio_thread.stop()

    def saveRecording(self,filename):
        self.audio_thread.saveAudio()
        self.video_thread.showFramesResume()

        subprocess.call('ffmpeg.exe -y -ac 2 -channel_layout stereo -i temp.wav -i temp.mp4 -pix_fmt yuv420p \"'+filename+'\"', shell=True)

        time.sleep(0.2)

        try:
            os.remove("temp.wav")
        except:
            print("Error : Cannot delete temporate Audio file")
        try:
            os.remove("temp.mp4")
        except:
            print("Error : Cannot delete temporate Video file")
            

    class VideoRecorder():
        def __init__(self, recorder, name, frameSize,isLive=True, fourcc="mp4v", camindex=0, fps=15):
            self.recorder = recorder
            self.device_index = camindex
            self.fps = fps                          
            self.fourcc = fourcc                    
            self.video_filename = name + ".mp4"
            self.video_frameSize = frameSize
            self.video_cap = cv2.VideoCapture(self.device_index)
            self.isLive = isLive
            self.reset(resetOut = False)
            self.face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.mouth_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')
            self.eye_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        def record(self):
            while self.open:
                ret, video_frame = self.video_cap.read()
                if ret:
                    video_frame = cv2.resize(video_frame, self.video_frameSize)
                    self.video_out.write(video_frame)
                    self.frame_counts += 1
                    self.duration += 1/(self.fps)


                    if self.isLive and self.frame_counts % 10 == 0:
    
                        faces = self.face_detection.detectMultiScale(video_frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
                        if len(faces) > 0:
                            faces = faces[0]
                            self.liveFace = faces
                            self.liveRectPos = (faces[0],faces[1],faces[0]+faces[2],faces[1]+faces[3])
                            self.liveFrameFace = faces

                            
                            eyes = self.eye_detection.detectMultiScale(video_frame,scaleFactor=1.1,minNeighbors=5)
                            mouth = self.mouth_detection.detectMultiScale(video_frame,scaleFactor=1.5,minNeighbors=11)

                            if len(eyes) >= 2:
                                self.liveEyeL = (eyes[0][0],eyes[0][1],eyes[0][0]+eyes[0][2],eyes[0][1]+eyes[0][3])
                                self.liveEyeR = (eyes[1][0],eyes[1][1],eyes[1][0]+eyes[1][2],eyes[1][1]+eyes[1][3])

                            if len(mouth) > 0:
                                self.liveMouth = (mouth[0][0],mouth[0][1],mouth[0][0]+mouth[0][2],mouth[0][1]+mouth[0][3])

                        self.liveFrameRecognition = video_frame
                        self.startRecognitionThread()
                    
                    while(not self.isLive and self.duration - self.recorder.audio_thread.duration >= 0.2 and self.recorder.audio_thread.open):
                        time.sleep(0.2)
                else:
                    break

            self.video_out.release()
            self.video_out = None

        def reset(self, resetOut = True):
            self.open=False          
            self.duration = 0
            self.video_writer = cv2.VideoWriter_fourcc(*(self.fourcc))
            if resetOut:
                self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.video_frameSize)
            self.frame_counts = 1
            self.start_time = time.time()
            self.liveRectPos = (0,0,0,0)
            self.liveRectEmotion = "NEUTRAL"
            self.recognitionThread = None
            self.liveFrameRecognition = None
            self.liveFrameFace = None
            self.liveMouth = (0,0,0,0)
            self.liveEyeL = (0,0,0,0)
            self.liveEyeR = (0,0,0,0)

        def stop(self):
            self.open=False
            self.stopRecognitionThread()

        def start(self):
            self.reset()
            self.open = True
            self.thread = threading.Thread(target=self.record)
            self.thread.start()

        def showFramesResume(self):
            frame_counts = self.frame_counts
            elapsed_time = time.time() - self.start_time
            recorded_fps = self.frame_counts / elapsed_time


        def recognize(self):
            self.liveRectEmotion = self.recorder.processing.process_image(self.liveFrameRecognition,self.liveFrameFace)
            #print("Video Live Emotion : ",self.liveRectEmotion)
            self.recognitionThread = None

        def stopRecognitionThread(self):
            self.recognitionThread = None

        def startRecognitionThread(self):
            if self.recognitionThread != None:
                return
            self.recognitionThread = threading.Thread(target=self.recognize)
            self.recognitionThread.start()
            

    class AudioRecorder():
        
        def __init__(self, recorder, filename, isLive=True,liveStopAfterSeconds=0.5, liveThreshold=10, rate=44100, fpb=1024, channels=1, audio_index=0):
            self.recorder = recorder
            self.rate = rate
            self.frames_per_buffer = fpb
            self.channels = channels
            self.format = pyaudio.paInt16
            self.audio_filename = filename + ".wav"
            self.audioIndex = audio_index
            self.isLive = isLive
            self.liveThreshold = liveThreshold
            self.liveStopAfterSeconds = liveStopAfterSeconds
            self.reset()
            self.recognitionThread = None
            self.liveEmotion = "NEUTRAL"
            self.liveText = ""


        def reset(self):
            self.open = False
            self.duration = 0
            self.audio_frames = []
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.audioIndex,
                frames_per_buffer = self.frames_per_buffer)
            self.lastSpike = time.time()
            self.liveHadSpike = False
            


        def record(self):
            self.stream.start_stream()
            t_start = time.time_ns()
            while self.open:
                try:
                    data = self.stream.read(self.frames_per_buffer)
                    self.duration += self.frames_per_buffer / self.rate
                    self.audio_frames.append(data)

                    if self.isLive and ( audioop.rms(data, 1) >= self.liveThreshold):
                        self.lastSpike = time.time()
                        self.liveHadSpike = True
                    elif self.isLive and self.liveHadSpike and time.time() - self.lastSpike >= self.liveStopAfterSeconds and len(self.audio_frames) > 0:
                        print("Checking audio for emotions")
                        # Check Audio for emotion
                        self.stream.stop_stream()
                        self.stream.close()
                        
                        self.saveAudio()

                        self.audio.terminate()
                        self.startRecognitionThread()
                        self.open = False
                        self.thread.stop()
                        self.thread = None

                        return
                        
                except Exception as e:
                    print('Audio Reading Error')
                while(not self.isLive and self.duration - self.recorder.video_thread.duration >= 0.5):
                    time.sleep(0.5)

            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

        def getAplitudes(self):
            return [audioop.rms(string, 1) for string in self.audio_frames]

        def stop(self):
            self.stopRecognitionThread()
            self.open = False

        def start(self):
            self.reset()
            self.open = True
            self.thread = threading.Thread(target=self.record)
            self.thread.start()

        def saveAudio(self):
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()


        def recognize(self):
            self.liveText = self.recorder.processing.speech_to_text_from_file(self.audio_filename)
            value = self.recorder.processing.process_audio_noSTT(self.liveText)
            print("Audio Live Emotion : ",value)
            self.recognitionThread = None
            self.start()
            self.liveEmotion = value

        def stopRecognitionThread(self):
            self.recognitionThread = None

        def startRecognitionThread(self):
            if self.recognitionThread != None:
                return
            self.recognitionThread = threading.Thread(target=self.recognize)
            self.recognitionThread.start()





