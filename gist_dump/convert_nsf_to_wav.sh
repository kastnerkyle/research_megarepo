ind=0
mkdir wavconvert
trap "exit" SIGINT
while read -rd $'\0' file; do
    echo $ind" : ${file}"
    echo $ind" : ${file}" >> file_lookup.txt
    write_file=wavconvert/$ind.wav
    write_file_tmp=wavconvert/$ind"_tmp.wav"
    #gst-launch-0.10 -v filesrc location="${file}" ! nsfdec ! pulsesink
    gst-launch-0.10 -v filesrc location="${file}" ! nsfdec ! audioconvert ! wavenc ! filesink location=$write_file 2>&1 >/dev/null &
    sleep 5 
    # this kill still lets it write out way more than it should write?
    # followup trim with sox
    jobs -p | xargs kill 2>&1 >/dev/null
    ffmpeg -i $write_file -acodec pcm_s16le -ac 1 -ar 16000 $write_file_tmp 2>&1 >/dev/null
    # Force trim to 20 secs
    sox $write_file_tmp $write_file trim 0 20
    #sox overwrites the way we want
    rm $write_file_tmp
    ind=$((ind + 1))
done < <(find  . -type f -name '*.nsf' -print0)
