rm ~/tfkdllib_dir/duration_rnn/*
rm outputs/*mid
rm outputs/*wav

runtime=$(date +%s)
mkdir -p outputs/$runtime

python duration_rnn.py
which_ckpt=$(ls ~/tfkdllib_dir/duration_rnn/*valid*ckpt | sort | tail -n 1)
python sample_duration_rnn.py $which_ckpt $runtime
for i in outputs/*mid; do
    ./timidifyit.sh $i
done
cp outputs/*wav ~/tfkdllib_dir/duration_rnn/
cp outputs/*mid ~/tfkdllib_dir/duration_rnn/
tar czf training_results_$runtime.tar.gz ~/tfkdllib_dir/duration_rnn
mv training_results_*tar.gz outputs/
mv outputs/*mid outputs/$runtime
mv outputs/*wav outputs/$runtime
