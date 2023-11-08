from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO
import math
import streamlit as st
import cvzone
import time 
import sounddevice
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import datetime
import soundfile as sf
import speech_recognition

model = YOLO("yolo-Weights/yolov8s.pt")
 # Initialize Streamlit state
if 'stop_clicked' not in st.session_state:
    st.session_state['stop_clicked'] = False

count = 0
 # object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"
               ]



def recording():
    freq = 44100
    duration = 10
    recording = None

    try:
         # Attempt to record audio
         recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    except sounddevice.PortAudioError as e:
        print(f"An error occurred with the audio device: {e}")
    #sd.wait()
    if recording is not None:
        write("recording0.wav", freq, recording)
        wv.write("recording1.wav", recording, freq, sampwidth=2)
 

# def recording(filename, duration=5, sample_rate=44100):
#     print("Recording audio...")
#     audio_data = []

#     with sf.SoundFile(filename, mode='w', samplerate=sample_rate, channels=1, format='wav') as file:
#         for _ in range(int(duration * sample_rate)):
#             audio_sample = np.random.randn()  # Replace with your audio capture logic
#             audio_data.append(audio_sample)
#             file.write(audio_sample)

#     print("Recording complete.")    
    
def overlay_image():
    imgBack = cv2.imread("captured_frame.png")
    imgFront = cv2.imread("awqaf.png", cv2.IMREAD_UNCHANGED)

    if imgBack is None:
        st.error("Error: 'captured_frame.png' not loaded properly.")
        st.stop()

    if imgFront is None:
        st.error("Error: 'awqaf.png' not loaded properly.")
        st.stop()

    imgResult = cvzone.overlayPNG(imgBack, imgFront, [2, 2])
    
    dt = datetime.datetime.now()
    current_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    
    org = [160, 50]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    cv2.putText(imgResult, current_time , org, font, fontScale, color, thickness)

    # Save the result image
    cv2.imwrite("result.png", imgResult)

def video_frame_callback(frame):
    
    global captured_frame, flag_capture

    img = frame.to_ndarray(format="bgr24") if not isinstance(frame, np.ndarray) else frame
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            cls = int(box.cls[0])
            if classNames[cls] == 'person':
            
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
        
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)
        
                # class name
                
                
               
                print("Class name -->", classNames[cls])
                
                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            
    cv2.imwrite('captured_frame.png', img)
                        
    return av.VideoFrame.from_ndarray(img, format="bgr24" )



# # Streamlit WebRTC streamer
# webrtc_streamer(key="example", video_frame_callback=video_frame_callback)


# Page layout
def main_page():
    st.title("Home")
    # Define RTC Configuration if needed
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


    # Streamlit WebRTC streamer
    webrtc_ctx = webrtc_streamer(key="example2",
                                 video_frame_callback=video_frame_callback,
                                 rtc_configuration=rtc_configuration,
                                 media_stream_constraints={"video": True, "audio": True},
                                 async_processing=True)

    # Stop button
    if st.button('Next'):
        st.session_state['stop_clicked'] = True
        st.experimental_rerun()

def second_page():
    st.image('result.png' , width=200)
    st.title("Khutba Page")
    st.write("بسم الله الرحمن الرحيم")
    time.sleep(5)
    # output_filename = "recorded_audio.wav"
    # recording(output_filename)
    recording()

    recognizer = speech_recognition.Recognizer()
    audio_file = "recording1.wav"  
    
    

    def process_audio(audio_data):
        global count
        try:
            text = recognizer.recognize_google(audio_data, language='ar')
            text = text.lower()
            
            st.write(text)
            
        except speech_recognition.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            count+=1
            print(count)
            
            
        except speech_recognition.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    with speech_recognition.AudioFile(audio_file) as source:
       # recognizer.adjust_for_ambient_noise(source)
        recognizer.adjust_for_ambient_noise(source, duration=5)

        while True:
            try:
                audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=7)
                process_audio(audio)
                # count +=1 
                if count== 10:
                    break
                
            except speech_recognition.WaitTimeoutError:
                print("End of Audio File")
                break
        

    

# App routing
if st.session_state['stop_clicked']:
    
    overlay_image()
    time.sleep(5)
    second_page()
    
    
else:
    main_page()

