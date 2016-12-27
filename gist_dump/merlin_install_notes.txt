% Assumptions: vctk is in ~/vctk/VCTK-Corpus.tar.gz
% htk is in ~/htk/HTK-3.4.1.tar.gz
% festival is in ~/festival/festival-2.4-release.tar.gz
% speech_tools is in ~/speech_tools/speech_tools-2.4-release.tar.gz
% sptk is in ~/sptk
% extrafiles from http://festvox.org/packed/festival/2.4/
% festlex_CMU.tar.gz
% festlex_POSTLEX.tar.gz
% festlex_OALD.tar.gz
% festvox_cmu_us_slt_cg.tar.gz
% in ~/fest_extras

% To get VCTK...
% To Festival and HTK
% Festival
% http://festvox.org/packed/festival/2.4/festival-2.4-release.tar.gz
% speech_tools
% http://www.cstr.ed.ac.uk/downloads/festival/2.4/speech_tools-2.4-release.tar.gz

% HTK
% http://htk.eng.cam.ac.uk/download.shtml 
% Need a username and password
% mkdir ~/htk
% mkdir ~/festival
% mkdir ~/speech_tools
% mkdir ~/vctk

% FestVox
% http://festvox.org/festvox-2.7/festvox-2.7.0-release.tar.gz

cd ~/speech_tools
tar xzf speech_tools-2.4-release.tar.gz
% ~/speech_tools/speech_tools
cd speech_tools
./configure && make -j 4
% speech_tools are in ~/speech_tools/speech_tools/bin
cd ~/src/merlin/tools
ln -s ~/speech_tools/speech_tools .

cd ~/festival
ln -s ~/speech_tools/speech_tools .
tar xzf festival-2.4-release.tar.gz
cd festival
% multicore make -j 4 gives errors? wtf
./configure && make


cd ~/festvox
tar xzf festvox-2.7.0-release.tar.gz 
cd festvox
ESTDIR=~/speech_tools/speech_tools ./configure && make
cd ~/src/merlin/tools
ln -s ~/festvox/festvox .

cd ~/htk
tar xzf HTK-3.4.1.tar.gz
cd htk
PATCH IT FROM http://hts.sp.nitech.ac.jp/?Download#k7c50de8
sudo apt-get install g++-multilib
./configure --disable-hlmtools --disable-hslab && make
% files in ~/htk/htk/

cd ~/sptk
tar xzf SPTK-3.9.tar.gz
sudo apt-get install csh
cd SPTK-3.9
mkdir out && ./configure --prefix=`pwd`/out && make && make install

cd ~/fest_extras
tar xzf festlex_CMU.tar.gz
tar xzf festlex_OALD.tar.gz
tar xzf festlex_POSLEX.tar.gz
tar xzf festvox_cmu_us_slt_cg.tar.gz
cp -pr festival ~/festival/festival 

cd ~/src/merlin/tools
ln -s ~/sptk/SPTK-3.9/out/bin SPTK-3.9

cd ~/vctk
tar xzf VCTK-Corpus.tar.gz

hts_engine_API-1.10
./configure && make

./configure --with-fest-search-path=/Tmp/kastner/speech_synthesis/festival/examples --with-sptk-search-path=/Tmp/kastner/speech_synthesis/SPTK-3.9/out/bin --with-hts-search-path=/Tmp/kastner/speech_synthesis/htk/HTKTools --with-hts-engine-search-path=/Tmp/kastner/speech_synthesis/hts_engine_API-1.10/bin

cp -pr ~/src/merlin/egs/build_your_own_voice/s1 ~/src/merlin/egs/build_your_own_voice/kks1
cd ~/src/merlin/egs/build_your_own_voice/kks1

ln -s ~/festival/festival ~/src/merlin/tools/festival 
ln -s ~/htk/htk ~/src/merlin/tools/htk

bash ./01_setup.sh my_vctk_example

% alignment_script=~/src/merlin/misc/scripts/alignment/state_align/run_aligner.sh

cd ~/src/merlin/egs/build_your_own_voice/kks1/database
ln -s ~/vctk/VCTK-Corpus/txt .
ln -s ~/vctk/VCTK-Corpus/wav48 wav

cd ~/src/merlin/tools
bash compile_tools.sh
cd ~/src/merlin/egs/slt_arctic/s1
bash run_full_voice.sh

sudo apt-get install gawk

cd ~/src/merlin/misc/scripts/vocoder/world/
sed -e 's|merlin_dir=.*|merlin_dir="$HOME/src/merlin"|g' extract_features_for_merlin.sh > extract_features_edit.sh
bash extract_features_edit.sh

cd ~/src/merlin/misc/scripts/alignment/state_align/
mkdir ~/src/merlin/misc/scripts/alignment/state_align/prompt-lab/tmp
bash ./setup.sh

# convert symlinks to hardlinks

cd ~/src/merlin/tools
# Brutal... http://superuser.com/questions/560597/convert-symlinks-to-hard-links
find -type l -exec bash -c 'fi="$(readlink -m "$0")" && rm "$0" && cp -pfr $fi "$0"' {} \;

cd ~/src/merlin/misc/scripts/alignment/phone_align
echo 'export ESTDIR="~/src/merlin/tools/speech_tools"' > source.sh
echo 'export FESTDIR="~/src/merlin/tools/festival"' >> source.sh
echo 'export FESTVOXDIR="~/src/merlin/tools/festvox"' >> source.sh

bash setup.sh
source source.sh && bash ./run_aligner.sh config.cfg

cd ~/src/merlin/misc/scripts/alignment/state_align
echo 'export ESTDIR="~/src/merlin/tools/speech_tools"' > source.sh
echo 'export FESTDIR="~/src/merlin/tools/festival"' >> source.sh
echo 'export FESTVOXDIR="~/src/merlin/tools/festvox"' >> source.sh
echo 'export HTKDIR="~/src/merlin/tools/htk/HTKTools"'> source.sh

bash setup.sh
sed -i -e 's|HTKDIR=.*|HTKDIR='"$HOME"'/kkastner/src/merlin/tools/htk/HTKTools|g' config.cfg 
source source.sh && bash run_sligner.sh config.cfg
