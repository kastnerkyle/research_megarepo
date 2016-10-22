which_ckpt=$1
if [ -z "$1" ]; then
    echo "Usage: sampleit.sh /path/to/checkpoint.ckpt"
    echo "Will sample and put results in samples/ directory"
    exit
fi
runtime=$(date +%s)
mkdir samples/
python sample_duration_rnn.py $which_ckpt $runtime samples
timidity -Ow samples/*mid
