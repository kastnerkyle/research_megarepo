ffmpeg -i 111.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav
find ../mp3_symlinks/ -name *.mp3 -exec sh -c 'echo $(basename {} .mp3)' \;
find ../mp3_symlinks/ -name *.mp3 -exec sh -c 'ffmpeg -i {} -acodec pcm_s16le -ac 1 -ar 16000 $(basename {} .mp3).wav' \;