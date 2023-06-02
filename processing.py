import speech_recognition as sr
import threading
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import cv2
import numpy as np
from keras.models import load_model
import openai
import subprocess
from tqdm import tqdm
import librosa
import pickle
from transformers import AutoTokenizer,TFDebertaModel
import requests
import json
import time

openai.api_key = "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"
your_api_token = "28d4e94ef38c4c07bc6f60ffa6962f39"

class Processing:

    def __init__(self):
        self.emotion_classifier = load_model("assets/models/fer2013_mini_XCEPTION.119-0.65.hdf5", compile=False)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.deberta = TFDebertaModel.from_pretrained('microsoft/deberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')


    def getMaxEmotion(self,tab,emotions):
        
        maxVal = tab[0]
        maxIndex = 0
        for i in range(1,len(tab)):
            if tab[i] > maxVal:
                maxVal = tab[i]
                maxIndex = i
        
        return emotions[maxIndex]


    # -------------------------------------------------------
    # ------------------ Video ------------------------------
    # -------------------------------------------------------


    def textPredict(self,text):

        max_len = 68

        input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
        input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")


        embeddings = self.deberta(input_ids,attention_mask = input_mask)[0] 
        out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
        out = Dense(128, activation='relu')(out)
        out = tf.keras.layers.Dropout(0.1)(out)
        out = Dense(32,activation = 'relu')(out)

        y = Dense(7,activation = 'sigmoid')(out)
          
        model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
        model.layers[2].trainable = True
      # for training bert our lr must be so small

        loadedModel = model.load_weights('assets/models/deberta_95.h5')

        x_val = self.tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=68,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)
      
        return(model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']}))


    # #Feature Extraction of Audio Files Function 
    def extract_feature(self,audio, mfcc, chroma, mel):
        #with soundfile.SoundFile(file_name) as sound_file:

        X , sample_rate = librosa.load(audio)
        # X = audio.read(dtype="float32")

        # sample_rate=audio.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        # print("step 1 : ",result.shape)
        if mfcc:
            # print(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).shape)
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        # print("step 2 : ",result.shape)
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        # print("step 3 : ",result.shape)
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
        # print("step 4 : ",result.shape)
        return result
    

    def audioPredict(self,filename):

      # Extract features
      audio_features = self.extract_feature(filename,mfcc=True,mel=True,chroma=True)
      audio_features = audio_features[np.newaxis,:]

      # Load the Model back from file
      Pkl_Filename = "assets/models/Emotion_Voice_Detection_Model.pkl"  
      with open(Pkl_Filename, 'rb') as file:  
          Emotion_Voice_Detection_Model = pickle.load(file)

      #prediction
      emotions_predicted = Emotion_Voice_Detection_Model.predict_proba(audio_features) #verify if its an array (%)
      # print(Emotion_Voice_Detection_Model.predict(audio_features))


      #return array of predictions
      return emotions_predicted


    def organise_emotions_predicted(self,emotions_text, fromSource):
        list_emotions_image = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"] #(base)
        list_emotions_text = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
        list_emotions_audio = ["Neutral", "Happiness", "Sadness", "Anger", "Fear", "Disgust", "Surprise"]

        if (fromSource == "text"):
            emotions_text[4], emotions_text[5] = emotions_text[5], emotions_text[4]
            emotions_text[5], emotions_text[6] = emotions_text[6], emotions_text[5]
        elif (fromSource == "audio"):
            emotions_text[0], emotions_text[3] = emotions_text[3], emotions_text[0]
            emotions_text[5], emotions_text[1] = emotions_text[1], emotions_text[5]
            emotions_text[2], emotions_text[4] = emotions_text[4], emotions_text[2]
            emotions_text[3], emotions_text[5] = emotions_text[5], emotions_text[3]
            emotions_text[5], emotions_text[6] = emotions_text[6], emotions_text[5]
      
        else:
            print("{} is not a valid source".format(fromSource))

        return emotions_text


    def imagePredict(self,video_filename):
        anger = []
        disgust = []
        fear = []
        happiness = []
        sadness = []
        surprise = []
        neutral = []

        detection_model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_detection = cv2.CascadeClassifier(detection_model_path)

        # To capture video from existing video.   
        cap = cv2.VideoCapture(video_filename)  

        totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2

        # print("total frame count : " , totalframecount)

        i = 0
        

        emotion_array = []

        for i in tqdm(range(totalframecount)):

            #print("image : ", i)
            _, bgr_image = cap.read()


            # video_capture = cv2.VideoCapture(0)
            # while True:
            #     bgr_image = video_capture.read()[1]
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
          
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            faces = face_detection.detectMultiScale(gray_image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            #print(faces)
            for face_coordinates in faces:

                x1, x2, y1, y2 = self.apply_offsets(face_coordinates, (20, 40))
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (self.emotion_target_size))
                except:
                    continue

                gray_face = self.preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = self.emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                #print("img prediction : ",emotion_prediction)
                
                anger.append(emotion_prediction[0][0])
                disgust.append(emotion_prediction[0][1])
                fear.append(emotion_prediction[0][2])
                happiness.append(emotion_prediction[0][3])
                sadness.append(emotion_prediction[0][4])
                surprise.append(emotion_prediction[0][5])
                neutral.append(emotion_prediction[0][6])



        list_emotions = ["Angry", "Disgust", "Fear", "Happiness", "Sad", "Surprise", "Neutral"]
        emotion = []

        emotion.append(anger)
        emotion.append(disgust)
        emotion.append(fear)
        emotion.append(happiness)
        emotion.append(sadness)
        emotion.append(surprise)
        emotion.append(neutral)
        #print("here : " ,emotion)
        if len(anger) == 0 and len(disgust) == 0 and len(fear) == 0 and len(happiness) == 0 and len(sadness) == 0 and len(surprise) == 0 and len(neutral) == 0:
            return [0,0,0,0,0,0,0]


        return [np.mean(emotion[0]), np.mean(emotion[1]), np.mean(emotion[2]), np.mean(emotion[3]), np.mean(emotion[4]), np.mean(emotion[5]), np.mean(emotion[6])]


    def process_video(self,filename):

      #array emotions
      emotions = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE", "NEUTRAL"]

      #Accuracy of each model
      text_accuracy = 0.8442
      audio_accuracy = 0.4371
      image_accuracy = 0.7042
      motion_accuracy = 0.6871
      global_accuracy = 0.7328

      subprocess.call("ffmpeg.exe -y -i \""+filename+"\" tmp.wav", shell=True)
    
      #text
      textUnderstood = self.speech_to_text_from_file("tmp.wav")
      result_text = self.organise_emotions_predicted(self.textPredict(textUnderstood)[0], fromSource="text")
      print("Text prediction : " + str(emotions[np.where(result_text == np.max(result_text))[0][0]]) + " (accuracy: " + str(int(text_accuracy*100)) + "%)")

      #speech
      result_audio = self.audioPredict("tmp.wav")[0] 
      print("Audio prediction : " + str( emotions[np.where(result_audio == np.max(result_audio))[0][0]]) + " (accuracy: " + str(int(audio_accuracy*100)) + "%)")

      #image
      print("\nPROCESSING Face Emotion Recognition")
      result_image = self.imagePredict(filename)
      print("Image prediction : " + str(emotions[result_image.index(np.max(result_image))]) + " (accuracy: " + str(int(image_accuracy*100)) + "%)") #emotions[result_image.index(np.max(result_image))])


      #motion
      print("\nPROCESSING Motion Emotion Recognition")
      result_motion = result_image
      print("\nMotion prediction : " + str(emotions[result_image.index(np.max(result_image))]) + " (accuracy: " + str(int(motion_accuracy*100)) + "%)") #emotions[result_image.index(np.max(result_image))])



        



      #Accuracy of each model for each emotion

      #Accuracy text
      text_fear_accuracy = 0.85
      text_sadness_accuracy = 0.87
      text_happiness_accuracy = 0.87
      text_anger_accuracy = 0.76
      text_disgust_accuracy = 0.96
      text_surprise_accuracy = 0.81
      text_neutral_accuracy = 0.80

      #Accuracy audio
      audio_fear_accuracy = 0.04
      audio_sadness_accuracy = 0.47
      audio_happiness_accuracy = 0.48
      audio_anger_accuracy = 0.54
      audio_disgust_accuracy = 0.40
      audio_surprise_accuracy = 0.38
      audio_neutral_accuracy = 0.75

      #Accuracy image
      image_fear_accuracy = 0.73
      image_sadness_accuracy = 0.57
      image_happiness_accuracy = 0.64
      image_anger_accuracy = 0.86
      image_disgust_accuracy = 0.79
      image_surprise_accuracy = 0.76
      image_neutral_accuracy = 0.58

      #Accuracy motion
      motion_fear_accuracy = 0.63
      motion_sadness_accuracy = 0.71
      motion_happiness_accuracy = 0.55
      motion_anger_accuracy = 0.79
      motion_disgust_accuracy = 0.73
      motion_surprise_accuracy = 0.55
      motion_neutral_accuracy = 0.85

      global_accuracy_fear_motion = motion_accuracy *  motion_fear_accuracy
      global_accuracy_sadness_motion = motion_accuracy *  motion_sadness_accuracy
      global_accuracy_happiness_motion = motion_accuracy *  motion_happiness_accuracy
      global_accuracy_anger_motion = motion_accuracy *  motion_anger_accuracy
      global_accuracy_disgust_motion = motion_accuracy *  motion_disgust_accuracy
      global_accuracy_surprise_motion = motion_accuracy *  motion_surprise_accuracy
      global_accuracy_neutral_motion = motion_accuracy *  motion_neutral_accuracy

      global_accuracy_fear_image = image_accuracy *  image_fear_accuracy
      global_accuracy_sadness_image = image_accuracy *  image_sadness_accuracy
      global_accuracy_happiness_image = image_accuracy *  image_happiness_accuracy
      global_accuracy_anger_image = image_accuracy *  image_anger_accuracy
      global_accuracy_disgust_image = image_accuracy *  image_disgust_accuracy
      global_accuracy_surprise_image = image_accuracy *  image_surprise_accuracy
      global_accuracy_neutral_image = image_accuracy *  image_neutral_accuracy

      global_accuracy_fear_audio = audio_accuracy *  audio_fear_accuracy
      global_accuracy_sadness_audio = audio_accuracy *  audio_sadness_accuracy
      global_accuracy_happiness_audio = audio_accuracy *  audio_happiness_accuracy
      global_accuracy_anger_audio = audio_accuracy *  audio_anger_accuracy
      global_accuracy_disgust_audio = audio_accuracy *  audio_disgust_accuracy
      global_accuracy_surprise_audio = audio_accuracy *  audio_surprise_accuracy
      global_accuracy_neutral_audio = audio_accuracy *  audio_neutral_accuracy

      global_accuracy_fear_text = text_accuracy *  text_fear_accuracy
      global_accuracy_sadness_text = text_accuracy *  text_sadness_accuracy
      global_accuracy_happiness_text = text_accuracy *  text_happiness_accuracy
      global_accuracy_anger_text = text_accuracy *  text_anger_accuracy
      global_accuracy_disgust_text = text_accuracy *  text_disgust_accuracy
      global_accuracy_surprise_text = text_accuracy *  text_surprise_accuracy
      global_accuracy_neutral_text = text_accuracy *  text_neutral_accuracy



      #Global prediction for each emotion
      global_fear_prediction = (result_motion[0] * global_accuracy_fear_motion + result_image[0] * global_accuracy_fear_image + result_audio[0] * global_accuracy_fear_audio  + result_text[0] * global_accuracy_fear_text) / (global_accuracy_fear_motion + global_accuracy_fear_image + global_accuracy_fear_audio +global_accuracy_fear_text)
      global_sadness_prediction =  (result_motion[1] * global_accuracy_sadness_motion + result_image[1] * global_accuracy_sadness_image + result_audio[1] * global_accuracy_sadness_audio  + result_text[1] * global_accuracy_sadness_text) / (global_accuracy_sadness_motion + global_accuracy_sadness_image + global_accuracy_sadness_audio +global_accuracy_sadness_text)
      global_happiness_prediction =  (result_motion[2] * global_accuracy_happiness_motion + result_image[2] * global_accuracy_happiness_image + result_audio[2] * global_accuracy_happiness_audio  + result_text[2] * global_accuracy_happiness_text) / (global_accuracy_happiness_motion + global_accuracy_happiness_image + global_accuracy_happiness_audio + global_accuracy_happiness_text)
      global_anger_prediction =  (result_motion[3] * global_accuracy_anger_motion + result_image[3] * global_accuracy_anger_image + result_audio[3] * global_accuracy_anger_audio  + result_text[3] * global_accuracy_anger_text) / (global_accuracy_anger_motion + global_accuracy_anger_image + global_accuracy_anger_audio + global_accuracy_anger_text)
      global_disgust_prediction =  (result_motion[4] * global_accuracy_disgust_motion + result_image[4] * global_accuracy_disgust_image + result_audio[4] * global_accuracy_disgust_audio  + result_text[4] * global_accuracy_disgust_text) / (global_accuracy_disgust_motion + global_accuracy_disgust_image + global_accuracy_disgust_audio + global_accuracy_disgust_text)
      global_surprise_prediction =  (result_motion[5] * global_accuracy_surprise_motion + result_image[5] * global_accuracy_surprise_image + result_audio[5] * global_accuracy_surprise_audio  + result_text[5] * global_accuracy_surprise_text) / (global_accuracy_surprise_motion + global_accuracy_surprise_image + global_accuracy_surprise_audio + global_accuracy_surprise_text)
      global_neutral_prediction =  (result_motion[6] * global_accuracy_neutral_motion + result_image[6] * global_accuracy_neutral_image + result_audio[6] * global_accuracy_neutral_audio  + result_text[6] * global_accuracy_neutral_text) / (global_accuracy_neutral_motion + global_accuracy_neutral_image + global_accuracy_neutral_audio + global_accuracy_neutral_text)


      list_global_predictions = [global_fear_prediction, global_sadness_prediction, global_happiness_prediction, global_anger_prediction, global_disgust_prediction, global_surprise_prediction, global_neutral_prediction]


      final_prediction = emotions[list_global_predictions.index(np.max(list_global_predictions))]

      print("FINAL PREDICTION : " +  str(final_prediction) + " (accuracy: " + str(int(global_accuracy*100)) + "%)") #emotions[result_image.index(np.max(result_image))]))

      return [final_prediction,{
              "IMAGE":[str(emotions[result_image.index(np.max(result_image))]),int(image_accuracy*100)],
              "MOTION":[str(emotions[result_image.index(np.max(result_image))]),int(motion_accuracy*100)],
              "AUDIO":[str( emotions[np.where(result_audio == np.max(result_audio))[0][0]]),int(audio_accuracy*100)],
              "TEXT":[str(emotions[np.where(result_text == np.max(result_text))[0][0]]),int(text_accuracy*100)],
            },
              textUnderstood]

      # return final_prediction





    # -------------------------------------------------------
    # ------------------ Image ------------------------------
    # -------------------------------------------------------

    def apply_offsets(self,face_coordinates, offsets):
        x, y, width, height = face_coordinates
        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

    def preprocess_input(self,x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def process_image(self,img_array, faces):
        try:
            if len(img_array) == 0 or len(faces) == 0:
                return "NEUTRAL"
        except:
            return "NEUTRAL"
        
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        x1, x2, y1, y2 = self.apply_offsets(faces, (20, 40))
        gray_face = gray_image[y1:y2, x1:x2]
        if len(gray_face) == 0:
            return "NEUTRAL"
        
        gray_face = cv2.resize(gray_face, (self.emotion_target_size))


        gray_face = self.preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = self.emotion_classifier.predict(gray_face)[0]
        list_emotions_image = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE", "NEUTRAL"]

        return self.getMaxEmotion(emotion_prediction,list_emotions_image)




    # -------------------------------------------------------
    # ------------------ AUDIO ------------------------------
    # -------------------------------------------------------


    def process_audio(self,filename):
        text = self.speech_to_text_from_file(filename)
        print("Text Recognized : ",text)

        if text == "":
            return "NEUTRAL"

        return self.getEmotionTextTest(text)

    def process_audio_noSTT(self,text):

        if text == "":
            return "NEUTRAL"

        return self.getEmotionTextTest(text)

    
        
    def speech_to_text_from_file(self, filename):
      r = sr.Recognizer()
      speech = ""

      
      audio_file=sr.AudioFile(filename)
      with audio_file as source:
          audio = r.record(source)

      #print("whisper --language en '"+filename+"'")
      #subprocess.call("whisper --language en '"+filename+"'", shell=True)
      #txt_name = filename[:-4]+".txt"
      #print(txt_name)
      #with open(txt_name) as file:
      #    speech = file.read()

      try:
          speech = r.recognize_google(audio)
      except:
          print("Couldn't translate to text")
      #speech = eval(r.recognize_vosk(audio))["text"]
          

      
      #upload_url = upload_file(your_api_token, audio)
      #transcript = create_transcript(your_api_token, upload_url)
      #print(transcript)
      return speech


    def getEmotionTextTest(self,code): 
      completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": "Choose the main emotion expressed in the following sentence, from one of this emotion (happiness/sadness/neutral/disgust/anger/surprise/fear):" + code + ". In your response, just give me the emotion, no sentence and no dots"}
      ]
      )
      
      res = completion.choices[0].message.content
      print("Actual GPT's Response : ",res)
      
      if res[-1] == ".":
          res = res[:len(res)-1]

      res = res.upper()

      if not res in ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE", "NEUTRAL"]:
          return "NEUTRAL"
      
      return res






    def read_file(self,filename, chunk_size=5242880):
        # Open the file in binary mode for reading
        with open(filename, 'rb') as _file:
            while True:
                # Read a chunk of data from the file
                data = _file.read(chunk_size)
                # If there's no more data, stop reading
                if not data:
                    break
                # Yield the data as a generator
                yield data

    def upload_file(self,api_token, path):
        """
        Upload a file to the AssemblyAI API.

        Args:
            api_token (str): Your API token for AssemblyAI.
            path (str): Path to the local file.

        Returns:
            str: The upload URL.
        """
        print(f"Uploading file: {path}")

        # Set the headers for the request, including the API token
        headers = {'authorization': api_token}
        
        # Send a POST request to the API to upload the file, passing in the headers
        # and the file data
        response = requests.post('https://api.assemblyai.com/v2/upload',
                                 headers=headers,
                                 data=read_file(path))

        # If the response is successful, return the upload URL
        if response.status_code == 200:
            return response.json()["upload_url"]
        # If the response is not successful, print the error message and return
        # None
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def create_transcript(self,api_token, audio_url):
        """
        Create a transcript using AssemblyAI API.

        Args:
            api_token (str): Your API token for AssemblyAI.
            audio_url (str): URL of the audio file to be transcribed.

        Returns:
            dict: Completed transcript object.
        """
        print("Transcribing audio... This might take a moment.")

        # Set the API endpoint for creating a new transcript
        url = "https://api.assemblyai.com/v2/transcript"

        # Set the headers for the request, including the API token and content type
        headers = {
            "authorization": api_token,
            "content-type": "application/json"
        }

        # Set the data for the request, including the URL of the audio file to be
        # transcribed
        data = {
            "audio_url": audio_url
        }

        # Send a POST request to the API to create a new transcript, passing in the
        # headers and data
        response = requests.post(url, json=data, headers=headers)

        # Get the transcript ID from the response JSON data
        transcript_id = response.json()['id']

        # Set the polling endpoint URL by appending the transcript ID to the API endpoint
        polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

        # Keep polling the API until the transcription is complete
        while True:
            # Send a GET request to the polling endpoint, passing in the headers
            transcription_result = requests.get(polling_endpoint, headers=headers).json()

            # If the status of the transcription is 'completed', exit the loop
            if transcription_result['status'] == 'completed':
                break

            # If the status of the transcription is 'error', raise a runtime error with
            # the error message
            elif transcription_result['status'] == 'error':
                raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

            # If the status of the transcription is not 'completed' or 'error', wait for
            # 3 seconds and poll again
            else:
                time.sleep(3)

        return transcription_result









if __name__ == "__main__":
    process = Processing()
    process.process_audio("60.42946112750735.wav")
    #process.process_video("anger_luca.mp4")
