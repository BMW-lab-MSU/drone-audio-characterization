import sounddevice as sd
import soundfile as sf
import numpy as np
import csv
import os
import motor_control.motor_control as motor_control
from quickset_pan_tilt import controller, protocol

# Configuration
THROTTLE_VALUES = [10, 50, 100]  # Percent throttle
ANGLES = [0, 30, 60, 90]  # Degrees
RECORDINGS_PER_SETTING = 10  # Number of recordings per setting
DURATION = 1 # Duration of each recording in seconds
SAMPLE_RATE = 44100  # Audio sample rate in Hz
OUTPUT_DIR = "recordings"  # Directory to save recordings and metadata
propeller_type = "black plastic"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize pan-tilt mount controller
def setup_pan_tilt_controller(port):
    pan_tilt = controller.ControllerSerial(protocol.PTCR20(), port)

    # Make sure the pan-tilt mount is at (0,0) before starting; this
    # isn't very important, but we might as well start at a known location.
    pan_tilt.home()

    return pan_tilt

# Initialize drone controller
def setup_drone_controller(port):
    motor_control.connect(port)
    motor_control.arm()
    motor_control.set_throttle([0,0,0,0])

def set_tilt_angle(pan_tilt, angle):
    # Check for faults and clear any that exist
    hard_faults, soft_faults = pan_tilt.check_for_faults(pan_tilt.get_status())
    # print(hard_faults)
    while hard_faults:
        pan_tilt.fault_reset()
        hard_faults, soft_faults = pan_tilt.check_for_faults(pan_tilt.get_status())

    pan_tilt.move_absolute(0, angle)

# Function to record audio
def record_audio(duration, sample_rate):
    print(f"Recording for {duration} seconds...")
    return sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64', device=4)

# Function to save metadata
def save_metadata(metadata):
    metadata_file = os.path.join(OUTPUT_DIR, "metadata.csv")
    fieldnames = ['file_name', 'angle', 'throttle', 'recording_number', 'duration', 'sample_rate','propeller_type', 'motor_1_rpm', 'motor_2_rpm', 'motor_3_rpm', 'motor_4_rpm']

    # Append to the file if it exists, else create a new one with header
    file_exists = os.path.isfile(metadata_file)
    with open(metadata_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(metadata)

def main():

    print("--------------------------------------------")
    print("Setting up pan tilt mount:")
    pan_tilt = setup_pan_tilt_controller("/dev/ttyUSB0")
    print("--------------------------------------------")
    print("Setting up drone motor control:")
    setup_drone_controller("/dev/ttyACM0")
    print("--------------------------------------------")

    # Load existing metadata if available
    metadata_file = os.path.join(OUTPUT_DIR, "metadata.csv")
    metadata = []
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            metadata = [row for row in reader]

    try:
        for angle in ANGLES:
            # Check if we've already done this angle and resume if necessary
            angle_metadata = [item for item in metadata if item['angle'] == str(angle)]
            if angle_metadata:
                print(f"Resuming from angle {angle}...")
                continue  # Skip to the next angle if it's already done
            
            print(f"Setting up for angle {angle} degrees...")
            # Assume some physical adjustment happens here for the angle
            set_tilt_angle(pan_tilt,angle)
            input("Move the drone to the next position (press Enter when ready)...")  # Prompt user to move the drone

            for throttle in THROTTLE_VALUES:
                # Check if we've already done this throttle for the current angle and resume if necessary
                throttle_metadata = [item for item in metadata if item['throttle'] == str(throttle) and item['angle'] == str(angle)]
                if throttle_metadata:
                    print(f"Resuming from throttle {throttle}%...")
                    continue  # Skip to the next throttle if it's already done
                
                print(f"Setting throttle to {throttle}%...")
                motor_control.set_throttle([throttle, throttle, throttle, throttle])  # Uniform throttle for all motors
                motor_control.throw_out_old_telemetry()
                rpm_metadata = motor_control.get_rpm_telemetry()

                for i in range(RECORDINGS_PER_SETTING):
                    # Check if this recording number has already been done
                    recording_metadata = [item for item in metadata if item['recording_number'] == str(i + 1) and item['throttle'] == str(throttle) and item['angle'] == str(angle)]
                    if recording_metadata:
                        print(f"Resuming recording {i+1} for throttle {throttle}% at angle {angle}...")
                        continue  # Skip to the next recording if it's already done

                    print(f"Recording {i+1}/{RECORDINGS_PER_SETTING} for throttle {throttle}% at angle {angle} degrees...")
                    # Record audio
                    audio_data = record_audio(DURATION, SAMPLE_RATE)
                    sd.wait()  # Wait until recording is finished
                    
                    # File naming
                    file_name = f"angle_{angle}_throttle_{throttle}_recording_{i+1}.wav"
                    file_path = os.path.join(OUTPUT_DIR, file_name)
                    
                    # Save audio data
                    try:
                        sf.write(file_path, audio_data, SAMPLE_RATE)
                    except Exception as e:
                        print(f"Failed to save recording {file_name}: {e}")
                        continue

                    # Collect metadata
                    metadata_entry = {
                        "file_name": file_name,
                        "angle": angle,
                        "throttle": throttle,
                        "recording_number": i + 1,
                        "duration": DURATION,
                        "sample_rate": SAMPLE_RATE,
                        "propeller_type": propeller_type,
                        "motor_1_rpm": rpm_metadata[0],
                        "motor_2_rpm": rpm_metadata[1],
                        "motor_3_rpm": rpm_metadata[2],
                        "motor_4_rpm": rpm_metadata[3]
                    }
                    metadata.append(metadata_entry)
                    
                # Prompt user to move the drone to the next throttle setting before recording the next set
                input("Move the drone to the next throttle setting (press Enter when ready)...")

            # Save data after each angle setup to avoid losing progress
            save_metadata(metadata)
            print(f"Data for angle {angle} degrees saved.")

        # Save metadata after completing all setups
        save_metadata(metadata)
        print("Audio recordings and metadata saved successfully!")

    finally:
        # Turn off motors and disconnect
        motor_control.set_throttle([0, 0, 0, 0])
        print("Motors turned off.")

if __name__ == "__main__":
    main()