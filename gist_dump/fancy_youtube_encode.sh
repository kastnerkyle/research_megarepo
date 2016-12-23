# Based on example here https://trac.ffmpeg.org/wiki/Encode/YouTube
text=$(basename $1 .wav)
ffmpeg -i $1 -filter_complex \
"[0:a]avectorscope=s=640x518,pad=1280:720[vs]; \
[0:a]showspectrum=mode=separate:color=intensity:scale=cbrt:s=640x518[ss]; \
[0:a]showwaves=s=1280x202:mode=line[sw]; \
[vs][ss]overlay=w[bg]; \
[bg][sw]overlay=0:H-h,drawtext=fontfile=/usr/share/fonts/truetype/fonts-japanese-gothic.ttf:fontcolor=white:x=10:y=10:text=$text[out]" \
-map "[out]" -map 0:a -c:v libx264 -preset fast -crf 18 -c:a copy $text.mkv
