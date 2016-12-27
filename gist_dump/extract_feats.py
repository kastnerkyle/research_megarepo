from __future__ import print_function
import os
import shutil
import stat
import subprocess
import time
import numpy as np

# File to extract features (mostly) automatically using the merlin speech
# pipeline
# example tts_env.sh file , written out by installer script install_tts.py
# https://gist.github.com/kastnerkyle/001a58a58d090658ee5350cb6129f857
"""
export ESTDIR=/Tmp/kastner/speech_synthesis/speech_tools/
export FESTDIR=/Tmp/kastner/speech_synthesis/festival/
export FESTVOXDIR=/Tmp/kastner/speech_synthesis/festvox/
export VCTKDIR=/Tmp/kastner/vctk/VCTK-Corpus/
export HTKDIR=/Tmp/kastner/speech_synthesis/htk/
export SPTKDIR=/Tmp/kastner/speech_synthesis/SPTK-3.9/
export HTSENGINEDIR=/Tmp/kastner/speech_synthesis/hts_engine_API-1.10/
export HTSDEMODIR=/Tmp/kastner/speech_synthesis/HTS-demo_CMU-ARCTIC-SLT/
export HTSPATCHDIR=/Tmp/kastner/speech_synthesis/HTS-2.3_for_HTL-3.4.1/
export MERLINDIR=/Tmp/kastner/speech_synthesis/latest_features/merlin/
"""

# Not currently needed...
def subfolder_select(subfolders):
    r = [sf for sf in subfolders if sf == "p294"]
    if len(r) == 0:
        raise ValueError("Error: subfolder_select failed")
    return r

# Need to edit the conf...
def replace_conflines(conf, match, sub):
    replace = None
    for n, l in enumerate(conf):
        if l[:len(match)] == match:
            replace = n
            break
    conf[replace] = "%s: %s\n" % (match, sub)
    return conf

def replace_write(fpath, match, sub):
    with open(fpath, "r") as f:
        conf = f.readlines()
    conf = replace_conflines(conf, match, sub)

    with open(fpath, "w") as f:
        f.writelines(conf)

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass  # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

# Convenience function to reuse the defined env
def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=False):
    """
    Print and execute command on system
    """
    for line in execute(cmd, shell=shell):
        print(line, end="")

# from merlin
def load_binary_file(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    features = features[:(dimension * (features.size / dimension))]
    features = features.reshape((-1, dimension))
    return  features

def array_to_binary_file(data, output_file_name):
    data = np.array(data, 'float32')
    fid = open(output_file_name, 'wb')
    data.tofile(fid)
    fid.close()

def load_binary_file_frame(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    frame_number = features.size / dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return  features, frame_number


# Source the tts_env_script
env_script = "tts_env.sh"
if os.path.isfile(env_script):
    command = 'env -i bash -c "source %s && env"' % env_script
    for line in execute(command, shell=True):
        key, value = line.split("=")
        # remove newline
        value = value.strip()
        os.environ[key] = value
else:
    raise IOError("Cannot find file %s" % env_script)

festdir = os.environ["FESTDIR"]
festvoxdir = os.environ["FESTVOXDIR"]
estdir = os.environ["ESTDIR"]
sptkdir = os.environ["SPTKDIR"]
# generalize to more than VCTK when this is done...

vctkdir = os.environ["VCTKDIR"]
htkdir = os.environ["HTKDIR"]
merlindir = os.environ["MERLINDIR"]

def extract_intermediate_features(wav_path, txt_path):
    basedir = os.getcwd()
    latest_feature_dir = "latest_features"
    if not os.path.exists(latest_feature_dir):
        os.mkdir(latest_feature_dir)

    os.chdir(latest_feature_dir)
    latest_feature_dir = os.getcwd()

    if not os.path.exists("merlin"):
        clone_cmd = "git clone https://github.com/kastnerkyle/merlin"
        pe(clone_cmd, shell=True)

    os.chdir("merlin")
    merlin_dir = os.getcwd()
    os.chdir("egs/build_your_own_voice/s1")
    experiment_dir = os.getcwd()

    if not os.path.exists("database"):
        print("Creating database and copying in files")
        pe("bash 01_setup.sh my_new_voice", shell=True)

        # Copy in wav files
        wav_partial_path = wav_path #vctkdir + "wav48/"
        """
        subfolders = sorted(os.listdir(wav_partial_path))
        # only p294 for now...
        subfolders = subfolder_select(subfolders)
        os.chdir("database/wav")
        for sf in subfolders:
            wav_path = wav_partial_path + sf + "/*.wav"
            pe("cp %s ." % wav_path, shell=True)
        """
        to_copy = os.listdir(wav_partial_path)
        if len([tc for tc in to_copy if tc[-4:] == ".wav"]) == 0:
            raise IOError("Unable to find any wav files in %s, make sure the filenames end in .wav!" % wav_partial_path)
        os.chdir("database/wav")
        wav_match_path = wav_partial_path + "/*.wav"
        pe("cp %s ." % wav_match_path, shell=True)

        # downsample the files
        convert = estdir + "bin/ch_wave $i -o tmp_$i -itype wav -otype wav -F 16000 -f 48000"
        pe("for i in *.wav; do echo %s; %s; mv tmp_$i $i; done" % (convert, convert), shell=True)

        os.chdir(experiment_dir)
        txt_partial_path = txt_path #vctkdir + "txt/"
        """
        subfolders = sorted(os.listdir(txt_partial_path))
        # only p294 for now...
        subfolders = subfolder_select(subfolders)
        os.chdir("database/txt")
        for sf in subfolders:
            txt_path = txt_partial_path + sf + "/*.txt"
            pe("cp %s ." % txt_path, shell=True)
        """
        os.chdir("database/txt")
        to_copy = os.listdir(txt_partial_path)
        if len([tc for tc in to_copy if tc[-4:] == ".txt"]) == 0:
            raise IOError("Unable to find any txt files in %s. Be sure the filenames end in .txt!" % txt_partial_path)
        txt_match_path = txt_partial_path + "/*.txt"
        pe("cp %s ." % txt_match_path, shell=True)

    do_state_align = False
    if do_state_align:
        raise ValueError("Replace these lies with something that points at the right place")
        os.chdir(merlin_dir)
        os.chdir("misc/scripts/alignment/state_align")
        pe("bash setup.sh", shell=True)

        with open("config.cfg", "r") as f:
            config_lines = f.readlines()

        # replace FESTDIR with the correct path
        festdir_replace_line = None
        for n, l in enumerate(config_lines):
            if "FESTDIR=" in l:
                festdir_replace_line = n
                break

        config_lines[festdir_replace_line] = "FESTDIR=%s\n" % festdir

        # replace HTKDIR with the correct path
        htkdir_replace_line = None
        for n, l in enumerate(config_lines):
            if "HTKDIR=" in l:
                htkdir_replace_line = n
                break

        config_lines[htkdir_replace_line] = "HTKDIR=%s\n" % htkdir

        with open("config.cfg", "w") as f:
            f.writelines(config_lines)

        pe("bash run_aligner.sh config.cfg", shell=True)
    else:
        os.chdir(merlin_dir)
        if not os.path.exists("misc/scripts/alignment/phone_align/full-context-labels/full"):
            os.chdir("misc/scripts/alignment/phone_align")
            pe("bash setup.sh", shell=True)

            with open("config.cfg", "r") as f:
                config_lines = f.readlines()

            # replace ESTDIR with the correct path
            estdir_replace_line = None
            for n, l in enumerate(config_lines):
                if "ESTDIR=" in l and l[0] == "E":
                    estdir_replace_line = n
                    break

            config_lines[estdir_replace_line] = "ESTDIR=%s\n" % estdir

            # replace FESTDIR with the correct path
            festdir_replace_line = None
            for n, l in enumerate(config_lines):
                # EST/FEST
                if "FESTDIR=" in l and l[0] == "F":
                    festdir_replace_line = n
                    break

            config_lines[festdir_replace_line] = "FESTDIR=%s\n" % festdir

            # replace FESTVOXDIR with the correct path
            festvoxdir_replace_line = None
            for n, l in enumerate(config_lines):
                if "FESTVOXDIR=" in l:
                    festvoxdir_replace_line = n
                    break

            config_lines[festvoxdir_replace_line] = "FESTVOXDIR=%s\n" % festvoxdir

            with open("config.cfg", "w") as f:
                f.writelines(config_lines)

            with open("run_aligner.sh", "r") as f:
                run_aligner_lines = f.readlines()

            replace_line = None
            for n, l in enumerate(run_aligner_lines):
                if "cp ../cmuarctic.data" in l:
                    replace_line = n
                    break

            run_aligner_lines[replace_line] = "cp ../txt.done.data etc/txt.done.data\n"

            # Make the txt.done.data file
            def format_info_tup(info_tup):
                return "( " + str(info_tup[0]) + ' "' + info_tup[1] + '" )\n'

            # Now we need to get the text info
            txt_partial_path = txt_path # vctkdir + "txt/"
            cwd = os.getcwd()
            out_path = "txt.done.data"
            out_file = open(out_path, "w")
            """
            subfolders = sorted(os.listdir(txt_partial_path))
            # TODO: Avoid this truncation and have an option to select subfolder(s)...
            subfolders = subfolder_select(subfolders)

            txt_ids = []
            for sf in subfolders:
                print("Processing subfolder %s" % sf)
                txt_sf_path = txt_partial_path + sf + "/"
                for txtpath in os.listdir(txt_sf_path):
                    full_txtpath = txt_sf_path + txtpath
                    with open(full_txtpath, 'r') as f:
                        r = f.readlines()
                        assert len(r) == 1
                        # remove txt extension
                        name = txtpath.split(".")[0]
                        text = r[0].strip()
                        info_tup = (name, text)
                        txt_ids.append(name)
                        out_file.writelines(format_info_tup(info_tup))
            """
            txt_ids = []
            txt_l_path = txt_partial_path
            for txtpath in os.listdir(txt_l_path):
                print("Processing %s" % txtpath)
                full_txtpath = txt_l_path + txtpath
                with open(full_txtpath, 'r') as f:
                    r = f.readlines()
                    assert len(r) == 1
                    # remove txt extension
                    name = txtpath.split(".")[0]
                    text = r[0].strip()
                    info_tup = (name, text)
                    txt_ids.append(name)
                    out_file.writelines(format_info_tup(info_tup))
            out_file.close()
            pe("cp %s %s/txt.done.data" % (out_path, latest_feature_dir),
               shell=True)
            os.chdir(cwd)

            replace_line = None
            for n, l in enumerate(run_aligner_lines):
                if "cp ../slt_wav/*.wav" in l:
                    replace_line = n
                    break

            run_aligner_lines[replace_line] = "cp ../wav/*.wav wav\n"

            # Put wav file in the correct place
            wav_partial_path = experiment_dir + "/database/wav"
            """
            subfolders = sorted(os.listdir(wav_partial_path))
            """
            if not os.path.exists("wav"):
                os.mkdir("wav")
            cwd = os.getcwd()
            os.chdir("wav")
            """
            for sf in subfolders:
                wav_path = wav_partial_path + "/*.wav"
                pe("cp %s ." % wav_path, shell=True)
            """
            wav_match_path = wav_partial_path + "/*.wav"
            pe("cp %s ." % wav_match_path, shell=True)
            os.chdir(cwd)

            replace_line = None
            for n, l in enumerate(run_aligner_lines):
                if "cat cmuarctic.data |" in l:
                    replace_line = n
                    break

            run_aligner_lines[replace_line] = 'cat etc/txt.done.data | cut -d " " -f 2 > file_id_list.scp\n'

            with open("edit_run_aligner.sh", "w") as f:
                f.writelines(run_aligner_lines)

            pe("bash edit_run_aligner.sh config.cfg", shell=True)


    # compile vocoder
    os.chdir(merlin_dir)
    os.chdir("tools")
    if not os.path.exists("SPTK-3.9"):
        pe("bash compile_tools.sh", shell=True)

    # slt_arctic stuff
    os.chdir(merlin_dir)
    os.chdir("egs/slt_arctic/s1")

    global_config_file = "conf/global_settings.cfg"
    # This madness due to autogen configs...
    pe("bash scripts/setup.sh slt_arctic_full", shell=True)
    pe("bash scripts/prepare_config_files.sh %s" % global_config_file, shell=True)
    pe("bash scripts/prepare_config_files_for_synthesis.sh %s" % global_config_file, shell=True)
    # delete the setup lines from run_full_voice.sh
    pe("sed -i.bak -e '11d;12d;13d' run_full_voice.sh", shell=True)

    pushd = os.getcwd()
    os.chdir("conf")

    replace_write("acoustic_slt_arctic_full.conf", "subphone_feats", "coarse_coding")
    replace_write("acoustic_slt_arctic_full.conf", "dmgc", "60")
    replace_write("acoustic_slt_arctic_full.conf", "dbap", "1")
    # hack this to add an extra line in the config
    replace_write("acoustic_slt_arctic_full.conf", "dlf0", "1\ndo_MLPG: False")
    replace_write("acoustic_slt_arctic_full.conf", "TRAINDNN", "False")
    replace_write("acoustic_slt_arctic_full.conf", "DNNGEN", "False")
    replace_write("acoustic_slt_arctic_full.conf", "GENWAV", "False")
    replace_write("acoustic_slt_arctic_full.conf", "CALMCD", "False")

    replace_write("duration_slt_arctic_full.conf", "TRAINDNN", "False")
    replace_write("duration_slt_arctic_full.conf", "DNNGEN", "False")
    replace_write("duration_slt_arctic_full.conf", "CALMCD", "False")

    os.chdir(pushd)
    if not os.path.exists("slt_arctic_full_data"):
        pe("bash run_full_voice.sh", shell=True)

    pe("mv run_full_voice.sh.bak run_full_voice.sh", shell=True)

    os.chdir(merlin_dir)
    os.chdir("misc/scripts/vocoder/world")

    with open("extract_features_for_merlin.sh", "r") as f:
        ex_lines = f.readlines()

    ex_line_replace = None
    for n, l in enumerate(ex_lines):
        if "merlin_dir=" in l:
            ex_line_replace = n
            break

    ex_lines[ex_line_replace] = 'merlin_dir="%s"' % merlin_dir

    ex_line_replace = None
    for n, l in enumerate(ex_lines):
        if "wav_dir=" in l:
            ex_line_replace = n
            break

    ex_lines[ex_line_replace] = 'wav_dir="%s"' % (experiment_dir + "/database/wav")

    with open("edit_extract_features_for_merlin.sh", "w") as f:
        f.writelines(ex_lines)

    pe("bash edit_extract_features_for_merlin.sh", shell=True)

    os.chdir(basedir)
    os.chdir("latest_features")
    os.symlink(merlin_dir + "/egs/slt_arctic/s1/slt_arctic_full_data/feat", "audio_feat")
    os.symlink(merlin_dir + "/misc/scripts/alignment/phone_align/full-context-labels/full", "text_feat")

    print("Audio features in %s (and %s)" % (os.getcwd() + "/audio_feat", merlin_dir + "/egs/slt_arctic/s1/slt_arctic_full_data/feat"))
    print("Text features in %s (and %s)" % (os.getcwd() + "/text_feat", merlin_dir + "/misc/scripts/alignment/phone_align/full-context-labels/full"))
    os.chdir(basedir)


def extract_final_features():
    launchdir = os.getcwd()
    os.chdir("latest_features")
    basedir = os.path.abspath(os.getcwd()) + "/"
    text_files = os.listdir("text_feat")
    audio_files = os.listdir("audio_feat/bap")
    os.chdir("merlin/egs/build_your_own_voice/s1")
    expdir = os.getcwd()

    # make the file list
    file_list_base = "experiments/my_new_voice/duration_model/data/"
    if not os.path.exists(file_list_base):
        os.mkdir(file_list_base)

    file_list_path = file_list_base + "file_id_list_full.scp"
    with open(file_list_path, "w") as f:
        f.writelines([tef.split(".")[0] + "\n" for tef in text_files])

    if not os.path.exists(basedir + "file_id_list_full.scp"):
        os.symlink(os.path.abspath(file_list_path), os.path.abspath(basedir + "file_id_list_full.scp"))

    # make the file list
    file_list_base = "experiments/my_new_voice/acoustic_model/data/"
    if not os.path.exists(file_list_base):
        os.mkdir(file_list_base)

    file_list_path = file_list_base + "file_id_list_full.scp"
    with open(file_list_path, "w") as f:
        f.writelines([tef.split(".")[0] + "\n" for tef in text_files])

    if not os.path.exists(basedir + "file_id_list_full.scp"):
        os.symlink(os.path.abspath(file_list_path), os.path.abspath(basedir + "file_id_list_full.scp"))

    file_list_base = "experiments/my_new_voice/test_synthesis/"
    if not os.path.exists(file_list_base):
        os.mkdir(file_list_base)

    file_list_path = file_list_base + "test_id_list.scp"
    # debug with no test utterances
    with open(file_list_path, "w") as f:
        f.writelines(["\n",])
        #f.writelines([tef.split(".")[0] + "\n" for tef in text_files[:20]])

    if not os.path.exists(basedir + "test_id_list.scp"):
        os.symlink(os.path.abspath(file_list_path), os.path.abspath(basedir + "test_id_list.scp"))

    # now copy in the data - don't symlink due to possibilities of inplace
    # modification
    os.chdir(expdir)
    basedatadir = "experiments/my_new_voice/"
    os.chdir(basedatadir)

    labeldatadir = "duration_model/data/label_phone_align"
    if not os.path.exists(labeldatadir):
        os.mkdir(labeldatadir)

    # IT USES HTS STYLE LABELS
    copytree(basedir + "text_feat", labeldatadir)

    labeldatadir = "acoustic_model/data/label_phone_align"
    if not os.path.exists(labeldatadir):
        os.mkdir(labeldatadir)

    bapdatadir = "acoustic_model/data/bap"
    if not os.path.exists(bapdatadir):
        os.mkdir(bapdatadir)

    lf0datadir = "acoustic_model/data/lf0"
    if not os.path.exists(lf0datadir):
        os.mkdir(lf0datadir)

    mgcdatadir = "acoustic_model/data/mgc"
    if not os.path.exists(mgcdatadir):
        os.mkdir(mgcdatadir)

    # IT USES HTS STYLE LABELS
    copytree(basedir + "text_feat", labeldatadir)
    copytree(basedir + "audio_feat/bap", bapdatadir)
    copytree(basedir + "audio_feat/lf0", lf0datadir)
    copytree(basedir + "audio_feat/mgc", mgcdatadir)
    #pe("cp %s acoustic_model/data" % label_norm_HTS_420.dat)

    while len(os.listdir(mgcdatadir)) < len(os.listdir(basedir + "audio_feat/mgc")):
        print("waiting for mgc file copy to complete...")
        time.sleep(3)

    while len(os.listdir(lf0datadir)) < len(os.listdir(basedir + "audio_feat/lf0")):
        print("waiting for lf0 file copy to complete...")
        time.sleep(3)

    while len(os.listdir(bapdatadir)) < len(os.listdir(basedir + "audio_feat/bap")):
        print("waiting for bap file copy to complete...")
        time.sleep(3)

    num_audio_files = len(os.listdir(mgcdatadir))
    num_label_files = len(os.listdir(labeldatadir))
    num_files = min([num_audio_files, num_label_files])

    os.chdir(expdir)

    global_config_file="conf/global_settings.cfg"
    pe("bash scripts/prepare_config_files.sh %s" % global_config_file, shell=True)
    pe("bash scripts/prepare_config_files_for_synthesis.sh %s" % global_config_file, shell=True)

    replace_write("conf/acoustic_my_new_voice.conf", "train_file_number", str(num_files))
    replace_write("conf/acoustic_my_new_voice.conf", "valid_file_number", "0")
    replace_write("conf/acoustic_my_new_voice.conf", "test_file_number", "0")

    replace_write("conf/acoustic_my_new_voice.conf", "dmgc", "60")
    replace_write("conf/acoustic_my_new_voice.conf", "dbap", "1")
    # hack this to add an extra line in the config
    replace_write("conf/acoustic_my_new_voice.conf", "dlf0", "1\ndo_MLPG: False")

    replace_write("conf/acoustic_my_new_voice.conf", "TRAINDNN", "False")
    replace_write("conf/acoustic_my_new_voice.conf", "DNNGEN", "False")
    replace_write("conf/acoustic_my_new_voice.conf", "GENWAV", "False")
    replace_write("conf/acoustic_my_new_voice.conf", "CALMCD", "False")


    replace_write("conf/duration_my_new_voice.conf", "train_file_number", str(num_files))
    replace_write("conf/duration_my_new_voice.conf", "valid_file_number", "0")
    replace_write("conf/duration_my_new_voice.conf", "test_file_number", "0")

    replace_write("conf/duration_my_new_voice.conf", "TRAINDNN", "False")
    replace_write("conf/duration_my_new_voice.conf", "DNNGEN", "False")
    replace_write("conf/duration_my_new_voice.conf", "CALMCD", "False")

    pe("sed -i.bak -e '19,20d;30,39d' 03_run_merlin.sh", shell=True)
    pe("bash 03_run_merlin.sh 2>&1", shell=True)
    pe("mv 03_run_merlin.sh.bak 03_run_merlin.sh", shell=True)
    if not os.path.exists(basedir + "final_acoustic_data"):
        os.symlink(os.path.abspath("experiments/my_new_voice/acoustic_model/data"),
                                   basedir + "final_acoustic_data")
    if not os.path.exists(basedir + "final_duration_data"):
        os.symlink(os.path.abspath("experiments/my_new_voice/duration_model/data"),
                                   basedir + "final_duration_data")
    os.chdir(launchdir)


def save_numpy_features():
    n_ins = 420
    n_outs = 63  # 187

    feature_dir = "latest_features/"
    with open(feature_dir + "file_id_list_full.scp") as f:
        file_list = [l.strip() for l in f.readlines()]

    norm_info_dir = os.path.abspath("latest_features/norm_info/") + "/"
    if not os.path.exists(norm_info_dir):
        os.mkdir(norm_info_dir)

    acoustic_dir = os.path.abspath(feature_dir + "final_acoustic_data/") + "/"
    audio_norm_file = "norm_info_mgc_lf0_vuv_bap_%s_MVN.dat" % str(n_outs)
    audio_norm_source = acoustic_dir + audio_norm_file
    audio_norm_dest = norm_info_dir + audio_norm_file
    shutil.copy2(audio_norm_source, audio_norm_dest)

    with open(audio_norm_source) as fid:
        cmp_info = np.fromfile(fid, dtype=np.float32)
        cmp_info = cmp_info.reshape((2, -1))
    audio_norm = cmp_info

    label_norm_file = "label_norm_HTS_%s.dat" % n_ins
    label_norm_source = acoustic_dir + label_norm_file
    label_norm_dest = norm_info_dir + label_norm_file
    shutil.copy2(label_norm_source, label_norm_dest)

    with open(label_norm_source) as fid:
        cmp_info = np.fromfile(fid, dtype=np.float32)
        cmp_info = cmp_info.reshape((2, -1))
    label_norm = cmp_info

    speaker_set = [x[:4] for x in file_list]
    speaker_set = sorted(list(set(speaker_set)))
    speaker_dict = {x: i for i, x in enumerate(speaker_set)}
    speaker2code = speaker_dict
    code2speaker = {v: k for k, v in speaker2code.items()}

    text_file = feature_dir + 'txt.done.data'

    with open(text_file) as f:
        text_data = [l.strip() for l in f.readlines()]

    monophone_path = os.path.abspath("latest_features/monophones") + "/"
    if not os.path.exists(monophone_path):
        # Trailing "/" causes issues
        os.symlink(os.path.abspath("latest_features/merlin/misc/scripts/alignment/phone_align/cmu_us_slt_arctic/lab"), monophone_path[:-1])

    launchdir = os.getcwd()
    phone_files = {gl[:-4]: monophone_path + gl for gl in os.listdir(monophone_path)
                if gl[-4:] == ".lab"}

    text_ids = [td.split(" ")[1] for td in text_data]

    label_files_path = os.path.abspath("latest_features/final_acoustic_data/nn_no_silence_lab_420") + "/"
    # still has silence in it?
    #audio_files_path = os.path.abspath("latest_features/final_acoustic_data/nn_mgc_lf0_vuv_bap_63") + "/"
    audio_files_path = os.path.abspath("latest_features/final_acoustic_data/nn_norm_mgc_lf0_vuv_bap_63") + "/"
    label_files = {lf[:-4]: label_files_path + lf for lf in os.listdir(label_files_path) if lf[-4:] == ".lab"}
    audio_files = {af[:-4]: audio_files_path + af for af in os.listdir(audio_files_path) if af[-4:] == ".cmp"}

    error_files = [
        (i, x) for i, x in enumerate(text_ids) if x not in file_list]

    # Solve corrupted files issues
    for i, x in error_files:
        try:
            text_ids.remove(x)
        except ValueError:
            pass
        try:
            file_list.remove(x)
        except ValueError:
            pass
        text_data = [td for td in text_data if td.split(" ")[1] != x]

    text_utts = [td.split('"')[1] for td in text_data]
    text_tups = list(zip(text_ids, text_utts))
    text_lu = {k: v for k, v in text_tups}
    text_rlu = {v: k for k, v in text_lu.items()}

    assert len(text_tups) == len(file_list)
    assert sum([ti not in file_list for ti in text_ids]) == 0

    char_set = sorted(list(set(''.join(text_utts).lower())))
    char2code = {x: i for i, x in enumerate(char_set)}
    code2char = {v: k for k, v in char2code.items()}

    phone_set = tuple('sil',)
    for fid in file_list:
        with open(phone_files[fid]) as f:
            phonemes = [p.strip() for p in f.readlines()]
        phonemes = [x.strip().split(' ') for x in phonemes[1:]]
        durations, phonemes = zip(*[[float(x), z] for x, y, z in phonemes])
        phone_set = tuple(sorted(list(set(phone_set + phonemes))))
    phone2code = {x: i for i, x in enumerate(phone_set)}
    code2phone = {v: k for k, v in phone2code.items()}
    order = range(len(file_list))
    np.random.seed(1)
    np.random.shuffle(order)

    all_in_features = []
    all_out_features = []
    all_phonemes = []
    all_durations = []
    all_text = []
    all_ids = []
    for i, idx in enumerate(order):
        fid = file_list[idx]
        #if i % 100 == 0:
        #    print(i)
        in_features, lab_frame_number = load_binary_file_frame(
            label_files[fid], n_ins)
        out_features, out_frame_number = load_binary_file_frame(
            audio_files[fid], n_outs)

        #print(lab_frame_number)
        #print(out_frame_number)
        if lab_frame_number != out_frame_number:
            print("WARNING: misaligned frame size for %s, using min" % fid)
            mf = min(lab_frame_number, out_frame_number)
            in_features = in_features[:mf]
            out_features = out_features[:mf]

        with open(phone_files[fid]) as f:
            phonemes = f.readlines()

        phonemes = [x.strip().split(' ') for x in phonemes[1:]]
        durations, phonemes = zip(*[[float(x), z] for x, y, z in phonemes])

        # first non pause phoneme
        first_phoneme = next(
            k - 1 for k, x in enumerate(phonemes) if x != 'pau')

        last_phoneme = len(phonemes) - next(
            k - 1 for k, x in enumerate(phonemes[::-1]) if x != 'pau')

        phonemes = phonemes[first_phoneme:last_phoneme]
        durations = durations[first_phoneme:last_phoneme]

        assert phonemes[0] == 'pau'
        assert phonemes[-1] == 'pau'
        # assert 'pau' not in phonemes[1:-1]
        phonemes = phonemes[1:-1]

        durations = np.array(durations)
        durations = durations * 200
        durations = durations - durations[0]
        durations = durations[1:] - durations[:-1]
        durations = durations[:-1]
        durations = np.round(durations, 0).astype('int32')
        phonemes = np.array([phone2code[x] for x in phonemes], dtype='int32')
        all_in_features.append(in_features)
        all_out_features.append(out_features)
        all_phonemes.append(phonemes)
        all_durations.append(durations)
        all_text.append(text_lu[fid])
        all_ids.append(fid)

    assert len(all_in_features) == len(all_out_features)
    assert len(all_in_features) == len(all_phonemes)
    assert len(all_in_features) == len(all_durations)
    assert len(all_in_features) == len(all_text)
    assert len(all_in_features) == len(all_ids)

    if not os.path.exists("latest_features/numpy_features"):
        os.mkdir("latest_features/numpy_features")

    def oa(s_dict):
        a = []
        for i in range(max([int(k) for k in s_dict.keys()])):
            a.append(s_dict[i])
        return arr(a)

    def arr(s):
        return np.array(s)

    for i in range(len(all_ids)):
        print("Saving %s" % all_ids[i])
        save_dict = {"file_id": arr(all_ids[i]),
                    "phonemes": arr(all_phonemes[i]),
                    "durations": arr(all_durations[i]),
                    "text": arr(all_text[i]),
                    "text_features": arr(all_in_features[i]),
                    "text_norminfo": label_norm,
                    "audio_features": arr(all_out_features[i]),
                    "audio_norminfo": audio_norm,
                    "mgc_extent": arr(60),
                    "lf0_idx": arr(60),
                    "vuv_idx": arr(61),
                    "bap_idx": arr(62),
                    "code2phone": oa(code2phone),
                    "code2char": oa(code2char),
                    "code2speaker": oa(code2speaker),
                    }

        np.savez_compressed("latest_features/numpy_features/%s.npz" % all_ids[i],
                            **save_dict)


def generate_merlin_wav(
        data, gen_dir, file_basename=None, #norm_info_file,
        do_post_filtering=True, mgc_dim=60, fl=1024, sr=16000):
    # Made from Jose's code and Merlin
    gen_dir = os.path.abspath(gen_dir) + "/"
    if file_basename is None:
        base = "tmp_gen_wav"
    else:
        base = file_basename
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)

    file_name = os.path.join(gen_dir, base + ".cmp")
    """
    fid = open(norm_info_file, 'rb')
    cmp_info = numpy.fromfile(fid, dtype=numpy.float32)
    fid.close()
    cmp_info = cmp_info.reshape((2, -1))
    cmp_mean = cmp_info[0, ]
    cmp_std = cmp_info[1, ]

    data = data * cmp_std + cmp_mean
    """

    array_to_binary_file(data, file_name)
    # This code was adapted from Merlin. All licenses apply

    out_dimension_dict = {'bap': 1, 'lf0': 1, 'mgc': 60, 'vuv': 1}
    stream_start_index = {}
    file_extension_dict = {
        'mgc': '.mgc', 'bap': '.bap', 'lf0': '.lf0',
        'dur': '.dur', 'cmp': '.cmp'}
    gen_wav_features = ['mgc', 'lf0', 'bap']

    dimension_index = 0
    for feature_name in out_dimension_dict.keys():
        stream_start_index[feature_name] = dimension_index
        dimension_index += out_dimension_dict[feature_name]

    dir_name = os.path.dirname(file_name)
    file_id = os.path.splitext(os.path.basename(file_name))[0]
    features, frame_number = load_binary_file_frame(file_name, 63)

    for feature_name in gen_wav_features:

        current_features = features[
            :, stream_start_index[feature_name]:
            stream_start_index[feature_name] +
            out_dimension_dict[feature_name]]

        gen_features = current_features

        if feature_name in ['lf0', 'F0']:
            if 'vuv' in stream_start_index.keys():
                vuv_feature = features[
                    :, stream_start_index['vuv']:stream_start_index['vuv'] + 1]

                for i in range(frame_number):
                    if vuv_feature[i, 0] < 0.5:
                        gen_features[i, 0] = -1.0e+10  # self.inf_float

        new_file_name = os.path.join(
            dir_name, file_id + file_extension_dict[feature_name])

        array_to_binary_file(gen_features, new_file_name)

    pf_coef = 1.4
    fw_alpha = 0.58
    co_coef = 511

    sptkdir = merlindir + "tools/bin/SPTK-3.9/"
    #sptkdir = os.path.abspath("latest_features/merlin/tools/bin/SPTK-3.9") + "/"
    sptk_path = {
        'SOPR': sptkdir + 'sopr',
        'FREQT': sptkdir + 'freqt',
        'VSTAT': sptkdir + 'vstat',
        'MGC2SP': sptkdir + 'mgc2sp',
        'MERGE': sptkdir + 'merge',
        'BCP': sptkdir + 'bcp',
        'MC2B': sptkdir + 'mc2b',
        'C2ACR': sptkdir + 'c2acr',
        'MLPG': sptkdir + 'mlpg',
        'VOPR': sptkdir + 'vopr',
        'B2MC': sptkdir + 'b2mc',
        'X2X': sptkdir + 'x2x',
        'VSUM': sptkdir + 'vsum'}

    #worlddir = os.path.abspath("latest_features/merlin/tools/bin/WORLD") + "/"
    worlddir = merlindir + "tools/bin/WORLD/"
    world_path = {
        'ANALYSIS': worlddir + 'analysis',
        'SYNTHESIS': worlddir + 'synth'}

    fw_coef = fw_alpha
    fl_coef = fl

    files = {'sp': base + '.sp',
             'mgc': base + '.mgc',
             'f0': base + '.f0',
             'lf0': base + '.lf0',
             'ap': base + '.ap',
             'bap': base + '.bap',
             'wav': base + '.wav'}

    mgc_file_name = files['mgc']
    cur_dir = os.getcwd()
    os.chdir(gen_dir)

    #  post-filtering
    if do_post_filtering:
        line = "echo 1 1 "
        for i in range(2, mgc_dim):
            line = line + str(pf_coef) + " "

        pe(
            '{line} | {x2x} +af > {weight}'
            .format(
                line=line, x2x=sptk_path['X2X'],
                weight=os.path.join(gen_dir, 'weight')), shell=True)

        pe(
            '{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | '
            '{c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
            .format(
                freqt=sptk_path['FREQT'], order=mgc_dim - 1,
                fw=fw_coef, co=co_coef, mgc=files['mgc'],
                c2acr=sptk_path['C2ACR'], fl=fl_coef,
                base_r0=files['mgc'] + '_r0'), shell=True)

        pe(
            '{vopr} -m -n {order} < {mgc} {weight} | '
            '{freqt} -m {order} -a {fw} -M {co} -A 0 | '
            '{c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
            .format(
                vopr=sptk_path['VOPR'], order=mgc_dim - 1,
                mgc=files['mgc'],
                weight=os.path.join(gen_dir, 'weight'),
                freqt=sptk_path['FREQT'], fw=fw_coef, co=co_coef,
                c2acr=sptk_path['C2ACR'], fl=fl_coef,
                base_p_r0=files['mgc'] + '_p_r0'), shell=True)

        pe(
            '{vopr} -m -n {order} < {mgc} {weight} | '
            '{mc2b} -m {order} -a {fw} | '
            '{bcp} -n {order} -s 0 -e 0 > {base_b0}'
            .format(
                vopr=sptk_path['VOPR'], order=mgc_dim - 1,
                mgc=files['mgc'],
                weight=os.path.join(gen_dir, 'weight'),
                mc2b=sptk_path['MC2B'], fw=fw_coef,
                bcp=sptk_path['BCP'], base_b0=files['mgc'] + '_b0'), shell=True)

        pe(
            '{vopr} -d < {base_r0} {base_p_r0} | '
            '{sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
            .format(
                vopr=sptk_path['VOPR'],
                base_r0=files['mgc'] + '_r0',
                base_p_r0=files['mgc'] + '_p_r0',
                sopr=sptk_path['SOPR'],
                base_b0=files['mgc'] + '_b0',
                base_p_b0=files['mgc'] + '_p_b0'), shell=True)

        pe(
            '{vopr} -m -n {order} < {mgc} {weight} | '
            '{mc2b} -m {order} -a {fw} | '
            '{bcp} -n {order} -s 1 -e {order} | '
            '{merge} -n {order2} -s 0 -N 0 {base_p_b0} | '
            '{b2mc} -m {order} -a {fw} > {base_p_mgc}'
            .format(
                vopr=sptk_path['VOPR'], order=mgc_dim - 1,
                mgc=files['mgc'],
                weight=os.path.join(gen_dir, 'weight'),
                mc2b=sptk_path['MC2B'], fw=fw_coef,
                bcp=sptk_path['BCP'],
                merge=sptk_path['MERGE'], order2=mgc_dim - 2,
                base_p_b0=files['mgc'] + '_p_b0',
                b2mc=sptk_path['B2MC'],
                base_p_mgc=files['mgc'] + '_p_mgc'), shell=True)

        mgc_file_name = files['mgc'] + '_p_mgc'

    # Vocoder WORLD

    pe(
        '{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | '
        '{x2x} +fd > {f0}'
        .format(
            sopr=sptk_path['SOPR'], lf0=files['lf0'],
            x2x=sptk_path['X2X'], f0=files['f0']), shell=True)

    pe(
        '{sopr} -c 0 {bap} | {x2x} +fd > {ap}'.format(
            sopr=sptk_path['SOPR'], bap=files['bap'],
            x2x=sptk_path['X2X'], ap=files['ap']), shell=True)

    pe(
        '{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | '
        '{sopr} -d 32768.0 -P | {x2x} +fd > {sp}'.format(
            mgc2sp=sptk_path['MGC2SP'], alpha=fw_alpha,
            order=mgc_dim - 1, fl=fl, mgc=mgc_file_name,
            sopr=sptk_path['SOPR'], x2x=sptk_path['X2X'], sp=files['sp']),
    shell=True)

    pe(
        '{synworld} {fl} {sr} {f0} {sp} {ap} {wav}'.format(
            synworld=world_path['SYNTHESIS'], fl=fl, sr=sr,
            f0=files['f0'], sp=files['sp'], ap=files['ap'],
            wav=files['wav']),
    shell=True)

    pe(
        'rm -f {ap} {sp} {f0} {bap} {lf0} {mgc} {mgc}_b0 {mgc}_p_b0 '
        '{mgc}_p_mgc {mgc}_p_r0 {mgc}_r0 {cmp} weight'.format(
            ap=files['ap'], sp=files['sp'], f0=files['f0'],
            bap=files['bap'], lf0=files['lf0'], mgc=files['mgc'],
            cmp=base + '.cmp'),
    shell=True)
    os.chdir(cur_dir)


def get_reconstructions():
    features_dir = "latest_features/numpy_features/"
    for fp in os.listdir(features_dir)[:5]:
        print("Reconstructing %s" % fp)
        a = np.load(features_dir + fp)
        norm = a["audio_norminfo"]
        am = norm[0]
        astd = norm[1]
        af = a["audio_features"]
        r = af * astd + am
        generate_merlin_wav(r, "latest_features/gen",
                    file_basename=fp.split(".")[0],
                            do_post_filtering=False)

if __name__ == "__main__":
    launchdir = os.getcwd()
    import argparse
    parser = argparse.ArgumentParser(description="Extract audio and text features using speech synthesis toolkits including SPTK, HTS, HTK, and Merlin. Special thanks to Jose Sotelo and the Edinburgh Speech Synthesis team.",
                                     epilog="Example usage: python extract_features.py -w wav48/p294 -t txt/p294")
    parser.add_argument("--wav_dir", "-w",
                        help="filepath for directory of wav files",
                        required=True)
    parser.add_argument("--txt_dir", "-t",
                        help="filepath for directory of txt files",
                        required=True)
    args = parser.parse_args()

    wav_dir = os.path.abspath(args.wav_dir)
    txt_dir = os.path.abspath(args.txt_dir)
    if wav_dir[-1] != "/":
        wav_dir += "/"
    if txt_dir[-1] != "/":
        txt_dir += "/"

    """
    # handle .data files?
    import os

    with open("cmuarctic.data", "r") as f:
        lines = f.readlines()

    if not os.path.exists("txt"):
        os.mkdir("txt")

    for l in lines:
        ls = l.split('"')
        base = ls[0].split(" ")[1]
        txt = ls[-2].strip()
        with open("txt/%s.txt" % base, "w") as f:
            f.write("%s\n" % txt)
    """

    if not os.path.exists("latest_features"):
        extract_intermediate_features(wav_dir, txt_dir)
    elif os.path.exists("latest_features"):
        if not os.path.exists("latest_features/text_feat") and not os.path.exists("latest_features/audio_feat"):
            print("Redoing feature extraction")
            os.chdir("latest_features")
            if os.path.exists("merlin"):
                shutil.rmtree("merlin")
            if os.path.exists("text_feat"):
                os.remove("text_feat")
            if os.path.exists("audio_feat"):
                os.remove("audio_feat")
            extract_intermediate_features(wav_dir, txt_dir)
    if not os.path.exists("latest_features/final_duration_data") or not os.path.exists("latest_features/final_acoustic_data"):
        extract_final_features()
        print("Feature extraction complete!")
    if not os.path.exists("latest_features/numpy_features"):
        save_numpy_features()
    if not os.path.exists("latest_features/gen"):
        #pass
        get_reconstructions()
    # TODO: Add -clean argument
    print("All files generated, remove the directories to rerun")
