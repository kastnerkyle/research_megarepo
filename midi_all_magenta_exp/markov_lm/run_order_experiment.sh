# Needs abcMIDI http://ifdo.pugmarks.com/~seymour/runabc/abcMIDI-2016.05.05.zip
# and timidity
# assumes the built abcmidi utils are in the current directory
# Data is a text file that combines http://www.campin.me.uk/Aird/Aird-v{1..6}.abc 
# wget http://www.campin.me.uk/Aird/Aird-v{1..6}.abc 
# cat *abc > Aird-full.abc
# This file is then stored in a folder called local_data
all_results_fldr=markov_chain_experiments
mkdir -p $all_results_fldr
for i in 6 10 15 20; do
    for j in 0.5 1.0 20.0 1000.0; do
        python markov_lm.py data/Aird-full.abc $i $j | tee out.txt
        fldr=order_"$i"_temperature_"$j"
        mkdir -p $fldr
        mv out.txt $fldr
        pushd .
        cd $fldr
        # This will not work without abc2midi
        ~/abcmidi/abc2midi out.txt
        for f in *mid; do 
            ../timidifyit.sh $f
        done
        for f in *wav; do
            name=$fldr"_"$f
            mv $f $name
            mv $name "../$all_results_fldr/$name"
        done
        out_name=$fldr"_"out.txt
        mv out.txt $out_name
        mv $out_name "../$all_results_fldr"
        popd
        rm -r $fldr
    done
done
