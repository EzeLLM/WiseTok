from gtts import gTTS
import os

print('the program is definitely starting, or something!')
tts = gTTS(text='subreddit pull completed', lang='en-au')
print('tts created')
tts.save('subredditpullcompleted.mp3')
print('tts saved')
