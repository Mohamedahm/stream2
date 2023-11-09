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
from st_audiorec import st_audiorec
import speech_recognition as sr
import os
from deta import Deta

model = YOLO("yolo-Weights/yolov8s.pt")
 # Initialize Streamlit state
if 'stop_clicked' not in st.session_state:
    st.session_state['stop_clicked'] = False

project = Deta("d0l6q5gedtr_iUApirur9uViehT3GgPqDStDGyj7xWRF")
# deta = Deta(st.secrets["data_key "])
# db = deta.Base("myspacedata")
# Define the drive to store the files.
drive_name = 'myspacedrive'
drive = project.Drive(drive_name)
# uploaded_file = st.file_uploader("Choose a file")

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


def audiorec_demo_app():
    st.title('Streamlit Audio Recorder')

    wav_audio_data = st_audiorec()  # Assuming st_audiorec() is properly defined elsewhere

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        
        
    return wav_audio_data

def save_audio(wav_bytes):
    if wav_bytes is not None:
        # Save the WAV audio data to a file
        audio_file_name = 'temp_audio.wav'
        with open(audio_file_name, 'wb') as f:
            f.write(wav_bytes)
        return audio_file_name
    return None

def process_audio(file_name):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio_data = recognizer.record(source)  # Record the data from the entire file
        try:
            text = recognizer.recognize_google(audio_data, language='ar')
            return text.lower()
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        
# def audiorec_demo_app():

#     # TITLE and Creator information
#     st.title('streamlit audio recorder')
#     st.markdown('Implemented by '
#         '[Stefan Rummer](https://www.linkedin.com/in/stefanrmmr/) - '
#         'view project source code on '
                
#         '[GitHub](https://github.com/stefanrmmr/streamlit-audio-recorder)')
#     st.write('\n\n')

#     # TUTORIAL: How to use STREAMLIT AUDIO RECORDER?
#     # by calling this function an instance of the audio recorder is created
#     # once a recording is completed, audio data will be saved to wav_audio_data

#     wav_audio_data = st_audiorec() # tadaaaa! yes, that's it! :D

#     # add some spacing and informative messages
#     col_info, col_space = st.columns([0.57, 0.43])
#     with col_info:
#         st.write('\n')  # add vertical spacer
#         st.write('\n')  # add vertical spacer
#         st.write('The .wav audio data, as received in the backend Python code,'
#                  ' will be displayed below this message as soon as it has'
#                  ' been processed. [This informative message is not part of'
#                  ' the audio recorder and can be removed easily] 🎈')

#     if wav_audio_data is not None:
#         # display audio data as received on the Python side
#         col_playback, col_space = st.columns([0.58,0.42])
#         with col_playback:
#             st.audio(wav_audio_data, format='audio/wav')


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
    
    st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
                unsafe_allow_html=True)
    # Design change st.Audio to fixed height of 45 pixels
    st.markdown('''<style>.stAudio {height: 45px;}</style>''',
                unsafe_allow_html=True)
    # Design change hyperlink href link color
    st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
                unsafe_allow_html=True)  # darkmode
    st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
                unsafe_allow_html=True)  # lightmode

    #recording()
    # audiorec_demo_app()
    # recognizer = speech_recognition.Recognizer()
    # audio_file = "recording1.wav"  
    wav_bytes = audiorec_demo_app()  # Ensure that this function returns the audio bytes
    audio_file_name = save_audio(wav_bytes)
    
    if audio_file_name is not None:
        text = process_audio(audio_file_name)
        st.write(text)
        # db.put(audio_file_name)
        # Optionally, clean up the audio file after processing

        # If user attempts to upload a file.
        # bytes_data = audio_file_name.getvalue()
    
        # Show the image filename and image.
        # st.write(f'filename: {audio_file_name.name}')
        # st.audio(bytes_data)
    
        # Upload the image to deta using put with filename and data.
        # drive.put(audio_file_name.name, data=bytes_data)
        # os.remove(audio_file_name)
    
    

    # def process_audio(audio_data):
    #     global count
    #     try:
    #         text = recognizer.recognize_google(audio_data, language='ar')
    #         text = text.lower()
            
    #         st.write(text)
            
    #     except speech_recognition.UnknownValueError:
    #         print("Google Speech Recognition could not understand audio")
    #         count+=1
    #         print(count)
            
            
    #     except speech_recognition.RequestError as e:
    #         print("Could not request results from Google Speech Recognition service; {0}".format(e))

    # with speech_recognition.AudioFile(audio_file) as source:
    #    # recognizer.adjust_for_ambient_noise(source)
    #     recognizer.adjust_for_ambient_noise(source, duration=5)

    #     while True:
    #         try:
    #             audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=7)
    #             process_audio(audio)
    #             # count +=1 
    #             if count== 10:
    #                 break
                
    #         except speech_recognition.WaitTimeoutError:
    #             print("End of Audio File")
    #             break
        

    

# App routing
# if st.session_state['stop_clicked']:
    
#     overlay_image()
#     time.sleep(5)
#     second_page()
    
    
# else:
#     main_page()

