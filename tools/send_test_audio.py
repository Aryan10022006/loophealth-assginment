import wave
import struct
import requests

# Create 1 second of silence WAV
fname = 'test_silence.wav'
with wave.open(fname, 'w') as wf:
    nchannels = 1
    sampwidth = 2
    framerate = 16000
    nframes = framerate * 1
    comptype = 'NONE'
    compname = 'not compressed'
    wf.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    silence = (0).to_bytes(2, byteorder='little', signed=True)
    for _ in range(nframes):
        wf.writeframes(silence)

# POST to server
url = 'http://localhost:8000/chat'
files = {'audio': ('test_silence.wav', open(fname, 'rb'), 'audio/wav')}
print('Sending test audio...')
resp = requests.post(url, files=files)
print('Status:', resp.status_code)
try:
    print('Response headers:', resp.headers)
    if resp.status_code == 200:
        with open('response.wav', 'wb') as f:
            f.write(resp.content)
        print('Saved response.wav')
    else:
        print('Response body:', resp.text)
except Exception as e:
    print('Error reading response:', e)
