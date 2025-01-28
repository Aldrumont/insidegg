import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time
import subprocess
from threading import Thread
from gtts import gTTS
import pygame
import os
from pydub import AudioSegment
import tempfile
from evdev import InputDevice, categorize, ecodes, list_devices
import hashlib

# execution variables
play_pause_cmd = False
next_cmd = False
previous_cmd = False
running = False
canceled = False
bluetooth_connected = False

# GPIO setup for LEDs and motor
PINS = {"branco": 16, "vermelho": 20, "azul": 12, "verde": 21}
direction = 19
step = 13
EN_pin = 26
motor_steps = 5000 # Number of steps to turn the egg
motor_running = False

# Setup motor
mymotortest = RpiMotorLib.A4988Nema(direction, step, (40,40,40), "DRV8825")
GPIO.setmode(GPIO.BCM)
GPIO.setup(list(PINS.values()), GPIO.OUT)
GPIO.setup(EN_pin, GPIO.OUT)
GPIO.output(list(PINS.values()), GPIO.HIGH)

# Setup camera
picam2 = Picamera2()
main_config = {"size": (1920, 1080), "format": "XBGR8888"}
video_config = picam2.create_video_configuration(main=main_config)
picam2.configure(video_config)
picam2.set_controls({'LensPosition': 9, 'FrameRate': 30})
encoder = H264Encoder(10000000)

#setup bluetooth device
device_name = "NB Speaker"

# Function to initialize the dictionary with existing audio files
def initialize_audio_dict(directory="audios"):
    audio_dict = {}
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".mp3"):
                audio_dict[filename] = os.path.join(directory, filename)
    return audio_dict

# Function to update the dictionary when a new file is created
def update_audio_dict(audio_dict, text, speed, path):
    hash_object = hashlib.md5((text + str(speed)).encode())
    filename = hash_object.hexdigest() + '.mp3'
    audio_dict[filename] = path

def find_device_path(device_name):
    # List all input devices
    devices = [InputDevice(path) for path in list_devices()]
    for device in devices:
        if device_name :
            return device.path
    return None

# Function to speak the text
def speak(text, speed=1.5, audio_dict={}):
    hash_object = hashlib.md5((text + str(speed)).encode())
    filename = hash_object.hexdigest() + '.mp3'
    audio_path = os.path.join('audios', filename)

    # Check if the file already exists in the dictionary
    if filename not in audio_dict:
        # Convert the text to speech in Brazilian Portuguese
        tts = gTTS(str(text), lang='pt-br')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)

        # Load and modify the audio file
        sound = AudioSegment.from_mp3(temp_file.name)
        sound = speed_change(sound, speed)
        sound.export(audio_path, format="mp3")

        # Update the dictionary
        update_audio_dict(audio_dict, text, speed, audio_path)

        # Clean up temporary file
        os.unlink(temp_file.name)

    # Initialize pygame and play the sound
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

    # Wait for the sound to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def speed_change(sound, speed=1.0):
    # Change the speed of the sound
    return sound.speedup(playback_speed=speed)

def convert_h264_to_mp4(h264_file, mp4_file):
    #eg ffmpeg -framerate 24 -i test.h264 -c:v copy -f mp4 test.mp4
    # overwrite output file if it exists
    try:
        print('Converting {} to {}...'.format(h264_file, mp4_file))
        subprocess.run(['ffmpeg', '-y', '-framerate', '24', '-i', h264_file, '-c:v', 'copy', '-f', 'mp4', mp4_file], check=True)
        print('Done.')
        return True
    except Exception as e:
        print('Error converting {} to {}: {}'.format(h264_file, mp4_file, e))
        return False

# Function to turn up or down led by color
def turn_led(color, state, verbose=False):
    state = GPIO.LOW if state else GPIO.HIGH
    GPIO.output(PINS[color], state)
    if verbose:
        print(f"LED {color} turned {'on' if state == GPIO.LOW else 'off'}.")


def run_motor():
    # Function to start a thread that runs the motor. and update the global variable
    global motor_running
    
    motor_thread = Thread(target=mymotortest.motor_go, args=(False, "Full", motor_steps, 0.0005, True, 0.05))
    motor_thread.start()
    motor_running = True
    print("Motor started.")
    motor_thread.join()
    motor_running = False  
    print("Motor stopped.")

def get_last_video_index():
    # Function to get the index of the last video recorded
    try:
        last_video = subprocess.run(['ls', '-t', 'videos'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')[0]
        last_video_index = int(last_video.split('_')[1].split('.')[0])
        return last_video_index
    except Exception as e:
        print('Error getting last video index: {}'.format(e))
        return 0

def listener_bluetooth_device():
    global bluetooth_connected, play_pause_cmd, next_cmd, previous_cmd
    while True:
        device_path = find_device_path(device_name)
        if not device_path:
            print(f"Device '{device_name}' not found")
            speak(f"Dispositivo Bluetooth não encontrado. Aguardando...")
            bluetooth_connected = False
            time.sleep(5)
        else:
            bluetooth_connected = True
            device = InputDevice(device_path)
            print(f"Dispositivo Bluetooth conectado")
            speak("Dispositivo Bluetooth conectado. Aguardando comandos")
            break
    
    for event in device.read_loop():
        if event.type == ecodes.EV_KEY:
            categorized_event = categorize(event)
            keycode = categorized_event.keycode
            if categorized_event.keystate == 1: # Consider just pressed events
                if 'PAUSE' in keycode or 'PLAY' in keycode:
                    print(f"Key Play/Pause pressed")
                    # Toggle the play_pause_cmd variable
                    play_pause_cmd = not play_pause_cmd
                elif 'NEXT' in keycode:
                    print(f"Key Next pressed")
                    next_cmd = True
                elif 'PREVIOUS' in keycode:
                    print(f"Key Previous pressed")
                    previous_cmd = True

                
# Start the thread that listens to the bluetooth device
thread_bluetooth_device_listener = Thread(target=listener_bluetooth_device)
thread_bluetooth_device_listener.start()

while True:

    # Wait for the bluetooth device to be connected
    if not bluetooth_connected:
        print("Bluetooth device not connected. Waiting...")
        time.sleep(1)
        continue
    
    if previous_cmd:
        previous_cmd = False
        last_video_index = get_last_video_index()
        speak(f"O ultimo video gravado foi o {last_video_index}")
    
    if play_pause_cmd and not running:
        play_pause_cmd = False
        running = True
        # Get the index of the last video recorded
        last_video_index = get_last_video_index()
        current_index = last_video_index + 1
        speak(f"Iniciando o experimento. Video {current_index}")
        picam2.start_recording(encoder, 'test.h264')

        # Turn the egg and light up LEDs
        for color in PINS.keys():
            if play_pause_cmd:
                canceled = True
                speak("Experimento cancelado")
                break
            # speak(f"Iniciando a cor {color}")
            # Turn on LED
            turn_led(color, True, verbose=True)
            time.sleep(1.5)
            # Start the motor
            GPIO.output(EN_pin, GPIO.LOW)
            mymotortest.motor_go(False, "Full", motor_steps, 0.0005, True, 0.05)
            # Turn off LED
            turn_led(color, False, verbose=True)
            time.sleep(0.1)
            # Stop the motor
            GPIO.output(EN_pin, GPIO.HIGH)
            # speak(f"Finalizado a cor {color}")
                

        # Cleanup and stop recording
        GPIO.output(EN_pin, GPIO.HIGH)
        #GPIO.cleanup()
        picam2.stop_recording()
        # speak("Finalizando o experimento")
        # picam2.close()
        # speak(f"Salvando o video {current_index}")
        print("GPIO cleaned, video recording stopped, and resources released.")
        if not canceled:
            success = convert_h264_to_mp4('test.h264', f'videos/video_{current_index}.mp4')
            if success:
                print('Success!')
                speak(f"Video {current_index} salvo com sucesso")
            else:
                print('Failed.')
                speak(f"Erro ao salvar o video {current_index}")
        else:
            canceled = False
        speak("Experimento finalizado")
        running = False
