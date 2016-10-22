To use the code for training:

THEANO_FLAGS="floatX=float32,device=gpu" python fruitspeecher.py

After training some number of epochs (usually the full 2000), sample from it with:

THEANO_FLAGS="floatX=float32,device=gpu" python fruitspeecher.py -s model_checkpoint_1999.pkl -w "orange" -b .1 -sl 150

Options are documented with fruitspeecher.py -h but the big thing is adding bias, controlling the sample string,
or changing the fixed size sample length.
