from tkinter import *
from tkinter import filedialog as fd
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter.messagebox

from processing import *
from recorder import *

width, height = 800, 600
isLive = True
processing = Processing()
recorder = Recorder(width,height,isLive,processing)

replayVideo = False
replayFrame = 0
replayVideoFeed = None

emotions = {
    "HAPPINESS":{
        "name":"Happiness",
        "color":"green"
    },
    "FEAR":{
        "name":"Fear",
        "color":"purple"
    },
    "ANGER":{
        "name":"Anger",
        "color":"red"
    },
    "DISGUST":{
        "name":"Disgust",
        "color":"forest green"
    },
    "SADNESS":{
        "name":"Sadness",
        "color":"DarkOliveGreen"
    },
    "SURPRISE":{
        "name":"Surprise",
        "color":"DarkOrange"
    },
    "NEUTRAL":{
        "name":"Neutral",
        "color":"gray"
    },
}



# ------------------------------------------------------
# ------------------- App Creation ---------------------
# ------------------------------------------------------


app = Tk()
app.title("Yet Another Python Recorder")
app.resizable(False,False)
app.bind('<Escape>', lambda e: exitApp())



def exitApp():
    exit()

def loadImages():
    dic = {}
    path = "assets/img/"
    for photo in os.listdir(path):
        filePath = os.path.join(path, photo)
        if os.path.isfile(filePath) and filePath.endswith(".png"):
            dic[photo.split(".png")[0]] = PhotoImage(file=filePath)
    return dic

imgs = loadImages()

img_webcam = None

canvas = Canvas(app,width=width,height=height)
ID_webcam = canvas.create_image(width/2,height/2,image=img_webcam)
ID_liveRect = canvas.create_rectangle(0,0,0,0,outline="red")
ID_liveMouth = canvas.create_rectangle(0,0,0,0,outline="red")
ID_liveEyeL = canvas.create_rectangle(0,0,0,0,outline="red")
ID_liveEyeR = canvas.create_rectangle(0,0,0,0,outline="red")
ID_liveText = canvas.create_text(0,0,text=" ",fill="red",anchor="w")
ID_liveImg = canvas.create_image(0,0,image="")
ID_liveRectAudio = canvas.create_rectangle(0,55,150,80,fill="white",outline="black")
ID_liveTextAudio = canvas.create_text(5,70,text=" ",fill="red",anchor="w")
ID_recordButton = canvas.create_image(width/2,height-25,image=imgs["record_start"])
ID_stateImg = canvas.create_image(42,27,image=imgs["state_live"])
ID_savedText = None
ID_savedRect = None

canvas.pack(side=LEFT)

rightFrame = Frame(app)
rightFrame.pack(side=RIGHT)

videoResultLabel = Label(rightFrame,text="",anchor="w")
videoResultLabel.pack()

figure = plt.Figure(figsize=(3,2), dpi=100)
ax = figure.add_subplot(111)
chart_type = FigureCanvasTkAgg(figure, rightFrame)
chart_type.get_tk_widget().pack()


textUnderstood = Label(rightFrame,text="",anchor="w",width=60,wraplength=500)
textUnderstood.pack()


def refreshStateImg():
    global canvas
    if isLive:
        canvas.itemconfig(ID_stateImg,image=imgs["state_live"])
    else:
        canvas.itemconfig(ID_stateImg,image=imgs["state_record"])

def refreshRecordButtonGraphics():
    if recorder.recording():
        canvas.itemconfig(ID_recordButton,image=imgs["record_end"])
    else:
        canvas.itemconfig(ID_recordButton,image=imgs["record_start"])




# ------------------------------------------------------
# -------------- Control functions ---------------------
# ------------------------------------------------------

def loadFileForEmotionCheck():
    global videoResultLabel,textUnderstood
    
    print("Loading file...")
    filetypes = (
        ('Video Files', ('*.mov','*.mp4','*.webm')),
        ('All files', '*.*')
    )

    try:
        filename = fd.askopenfilename(
            title='Open a file',
            filetypes=filetypes)

        if filename != None and len(filename) > 0:
            print("File chosen : ",filename)
            res = processing.process_video(filename)
            print("Video Result : ",res[0])

            strTot = "Results : \n Emotion chosen : "+ emotions[res[0]]["name"]+"\n\n Emotions picked up :\n"

            for key in res[1]:
                strTot+= key + " : "+emotions[res[1][key][0]]["name"]+" ("+str(res[1][key][1])+"%)\n"

            videoResultLabel.config(text=strTot,anchor="w")
            textUnderstood.config(text="Text Understood : "+res[2],anchor="w")
            
            startReplay(filename)
    except:
        print("Error while loading file : ",filename)
        
def startReplay(filepath):
    global replayVideoFeed, replayVideo, replayFrame,ax,figure

    # Setup Video Replay
    replayVideo = True
    replayFrame = 0
    replayVideoFeed = cv2.VideoCapture(filepath)


    # Setup Audio Plot
    print("ffmpeg.exe -y -i "+filepath+" tmp.wav")
    subprocess.call("ffmpeg.exe -y -i \""+filepath+"\" tmp.wav", shell=True)

    audioFile = wave.open("tmp.wav", "rb")

    amplitudes = []

    data = audioFile.readframes(1024)
    while data:
        amplitudes.append(audioop.rms(data, 1))
        data = audioFile.readframes(1024)

    ax.cla()
    ax.plot(amplitudes)
    figure.canvas.draw()


    audioFile.close()
    try:
        os.remove("tmp.wav")
    except:
        print("Temp Audio Removal Failure")

    

def stopReplay():
    global replayVideo,replayVideoFeed,videoResultLabel,textUnderstood
    if replayVideo:
        replayVideo = False
        replayVideoFeed.release()
        replayVideoFeed = None
        videoResultLabel.config(text="")
        textUnderstood.config(text="")


def deleteSavedText():
    global ID_savedText,canvas, ID_savedRect
    canvas.delete(ID_savedText)
    canvas.delete(ID_savedRect)
    ID_savedText = None
    ID_savedRect = None


def startRecording():
    global recorder
    stopReplay()
    recorder.startRecording()
    recorder.audio_thread.liveText = ""
    recorder.audio_thread.persistentDuration = 0
    refreshRecordButtonGraphics()

def stopRecording():
    global recorder, ID_savedText,canvas, ID_savedRect
    recorder.stopRecording()
    refreshRecordButtonGraphics()
    if not isLive:
        types = [('MOV Files', '*.mov'),('All Files', '*.*')]
        filename = fd.asksaveasfile(filetypes = types, defaultextension = types)
        recorder.saveRecording(filename.name)
        
        ID_savedText = canvas.create_text(width/2,20,text="Video saved to : "+filename.name,fill="black")
        ID_savedRect=canvas.create_rectangle(canvas.bbox(ID_savedText),fill="white")

        filename.close()
        canvas.tag_lower(ID_savedRect,ID_savedText)
        canvas.after(2000, deleteSavedText)
    else:
        time.sleep(0.2)
        os.remove("temp.mp4")
        

def event_record(event):
    global canvas
    if recorder.recording():
        stopRecording()
    else:
        startRecording()


        

def refreshUI_replay():
    global replayFrame,canvas,img_webcam, replayVideoFeed

    replayVideoFeed.set(cv2.CAP_PROP_POS_FRAMES,replayFrame)
    replayFrame = ( replayFrame + 1 ) % replayVideoFeed.get(cv2.CAP_PROP_FRAME_COUNT)

    ret, frame = replayVideoFeed.read()
    if ret:
        video_frame = cv2.resize(frame, (width,height))
        opencv_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        img_webcam = ImageTk.PhotoImage(image=captured_image)
        canvas.itemconfig(ID_webcam,image=img_webcam)

        
        

def refreshUI_default():
    global canvas,img_webcam, ax, textUnderstood
    
    # Refresh Webcam image
    ret, frame = recorder.video_thread.video_cap.read()
    if ret:
        video_frame = cv2.resize(frame, (width,height))
        opencv_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        img_webcam = ImageTk.PhotoImage(image=captured_image)
        canvas.itemconfig(ID_webcam,image=img_webcam)

    # Refresh Live Rectangle
    x0,y0,x1,y1 = recorder.video_thread.liveRectPos 

    if isLive and recorder.recording():
        canvas.coords(ID_liveRect,x0,y0,x1,y1)
        emotionVideo = emotions[recorder.video_thread.liveRectEmotion]
        canvas.itemconfig(ID_liveText,text=emotionVideo["name"],fill=emotionVideo["color"])
        canvas.itemconfig(ID_liveRect,outline=emotionVideo["color"])
        canvas.coords(ID_liveText,x0,y1+10)
        canvas.itemconfig(ID_liveImg,image=imgs[recorder.video_thread.liveRectEmotion])
        canvas.coords(ID_liveImg,x1,y1+10)

        canvas.coords(ID_liveMouth,recorder.video_thread.liveMouth)
        canvas.coords(ID_liveEyeL,recorder.video_thread.liveEyeL)
        canvas.coords(ID_liveEyeR,recorder.video_thread.liveEyeR)

        
        emotionAudio = emotions[recorder.audio_thread.liveEmotion]
        canvas.itemconfig(ID_liveRectAudio,outline="black",fill="white")
        canvas.itemconfig(ID_liveTextAudio,text="Audio Emotion : "+emotionAudio["name"],fill=emotionAudio["color"])

        textUnderstood.config(text="Text Understood :\n"+recorder.audio_thread.liveText)
    else:
        canvas.coords(ID_liveRect,0,0,0,0)
        canvas.coords(ID_liveMouth,0,0,0,0)
        canvas.coords(ID_liveEyeL,0,0,0,0)
        canvas.coords(ID_liveEyeR,0,0,0,0)
        
        canvas.itemconfig(ID_liveImg,image="")
        canvas.itemconfig(ID_liveText,text=" ")
        canvas.itemconfig(ID_liveTextAudio,text=" ")
        canvas.itemconfig(ID_liveRectAudio,outline="",fill="")
        textUnderstood.config(text="")

    # Refresh Audio Graph
    ax.cla()
    ax.plot(recorder.audio_thread.getAplitudes())
    figure.canvas.draw()


def refreshUI():
    global canvas

    if replayVideo:
        refreshUI_replay()
    else:
        refreshUI_default()

    canvas.after(10, refreshUI)


def toggleRecordState():
    global isLive
    if recorder.recording():
        stopRecording()
    isLive = not isLive
    recorder.changeIsLive(isLive)
    refreshStateImg()
    stopReplay()


# ------------------------------------------------------
# ------------------ Menu Creation ---------------------
# ------------------------------------------------------

menu = Menu(app)
app.config(menu=menu)

editMenu = Menu(menu)
menu.add_cascade(label="File", menu=editMenu)
editMenu.add_command(label="Switch record/live", command=toggleRecordState)
editMenu.add_command(label="Load file", command=loadFileForEmotionCheck)


# ------------------------------------------------------
# --------------------- App Launch ---------------------
# ------------------------------------------------------

canvas.tag_bind(ID_recordButton,"<Button-1>",event_record)

refreshStateImg()
refreshUI()


app.mainloop()
