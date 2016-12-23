sudo add-apt-repository ppa:hugegreenbug/cmt2
sudo apt-get update
sudo apt-get install libevdevc libgestures xf86-input-cmt
sudo mv /usr/share/X11/xorg.conf.d/50-synaptics.conf /usr/share/X11/xorg.conf.d/50-synaptics.conf.old
sudo cp /usr/share/xf86-input-cmt/50-touchpad-cmt-samus.conf /usr/share/X11/xorg.conf.d/
