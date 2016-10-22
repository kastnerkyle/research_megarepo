which_ckpt=$(ls ~/tfkdllib_dir/duration_rnn/*valid*ckpt | sort | tail -n 1)
runtime=$(date +%s)
mkdir samples/
python sample_duration_rnn.py $which_ckpt $runtime samples
timidity -Ow samples/*mid
