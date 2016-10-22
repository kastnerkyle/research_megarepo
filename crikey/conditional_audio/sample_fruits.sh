#usage THEANO_FLAGS="blah" bash sample_fruits.sh python samplefile.py -s pickle_file
args=$@
for i in "apple" "banana" "peach" "pineapple" "lime" "kiwi" "orange"; do
    echo $i
    $@ -w $i -sl 140
done
for i in "banapple" "bappeach" "pineeach"; do
    $@ -w $i -sl 140
done
