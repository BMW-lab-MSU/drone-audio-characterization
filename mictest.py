import sounddevice as sd

devices = sd.query_devices()
print("Avaialable sound devices:")
print(devices)
