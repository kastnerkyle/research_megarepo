from __future__ import print_function
import subprocess
import shutil
import os
import stat
import time

# This script looks extremely defensive, but *should* let you rerun at
# any stage along the way. Also a lot of code repetition due to eventual support
# for "non-blob" install from something besides the magic kk_all_deps.tar.gz

# Contents of kk_all_deps.tar.gz
"""
all_deps/
all_deps/festlex_CMU.tar.gz
all_deps/festival-2.4-release.tar.gz
all_deps/HTK-3.4.1.tar.gz
all_deps/festlex_POSLEX.tar.gz
all_deps/HTS-2.3_for_HTK-3.4.1.tar.bz2
all_deps/VCTK-Corpus.tar.gz
all_deps/hts_engine_API-1.10.tar.gz
all_deps/festvox_cmu_us_slt_cg.tar.gz
all_deps/speech_tools-2.4-release.tar.gz
all_deps/festvox-2.7.0-release.tar.gz
all_deps/festlex_OALD.tar.gz
all_deps/SPTK-3.9.tar.gz
all_deps/HTS-demo_CMU-ARCTIC-SLT.tar.bz2
"""

# We are about to install a lot of things
# 2 primary directories inside base_dir
# all_deps/* will have all zipped dirs
# vctk/VCTK-Corpus will have all the data
# speech_synthesis/* will have a bunch of compiled C++ codebases
# we also set the environment appropriately and write out some helper scripts
starting_dir = os.getcwd()

base_install_dir = "/Tmp/kastner/"

base_synthesis_dir = base_install_dir + "speech_synthesis/"
base_vctk_dir = base_install_dir + "vctk/"

vctkdir = base_vctk_dir + "VCTK-Corpus/"
merlindir = base_synthesis_dir + "latest_features/merlin/"
estdir = base_synthesis_dir + "speech_tools/"
festdir = base_synthesis_dir + "festival/"
festvoxdir = base_synthesis_dir + "festvox/"
htkdir = base_synthesis_dir + "htk/"
sptkdir = base_synthesis_dir + "SPTK-3.9/"
htspatchdir = base_synthesis_dir + "HTS-2.3_for_HTL-3.4.1/"
htsenginedir = base_synthesis_dir + "hts_engine_API-1.10/"
htsdemodir = base_synthesis_dir + "HTS-demo_CMU-ARCTIC-SLT/"

# http://www.nguyenquyhy.com/2014/07/create-full-context-labels-for-hts/

env = os.environ.copy()

env["ESTDIR"] = estdir
env["FESTVOXDIR"] = festvoxdir
env["FESTDIR"] = festdir
env["VCTKDIR"] = vctkdir

def copytree(src, dst, symlinks = False, ignore = None):
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
        pass # lchmod not available
    elif os.path.isdir(s):
      copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)

# Convenience function to reuse the defined env
def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
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

# Setup all the directories
os.chdir(base_install_dir)
if not os.path.exists(base_synthesis_dir):
    os.mkdir(base_synthesis_dir)

if not os.path.exists(base_vctk_dir):
    os.mkdir(base_vctk_dir)

# Check for the big bundle
install_bundle = "kk_all_deps.tar.gz"
install_bundle_path = base_install_dir + install_bundle
if not os.path.exists(install_bundle):
    print("ERROR: Must have %s in %s" % (install_bundle, base_install_dir))
    raise IOError("Make sure the filepath %s has the right file" % install_bundle_path)

# Start unpacking things
os.chdir(base_synthesis_dir)

# create a temporary symlink to unzip
if not os.path.exists(base_synthesis_dir + install_bundle):
    os.symlink(install_bundle_path, base_synthesis_dir + install_bundle)

dep_dir = "all_deps/"
full_dep_dir = base_synthesis_dir + dep_dir
if not os.path.exists(full_dep_dir):
    print("Unpacking deps...")
    untar_cmd = ["tar", "xzf", install_bundle]
    pe(untar_cmd)

# Unpack vctk
# Install dir for vctk
os.chdir(base_install_dir)
if not os.path.exists(base_vctk_dir):
    os.mkdir(base_vctk_dir)

# symlink
os.chdir(base_vctk_dir)
vctk_pkg = "VCTK-Corpus.tar.gz"
vctk_pkg_path = base_vctk_dir + vctk_pkg
if not os.path.exists(vctk_pkg_path):
    os.symlink(base_synthesis_dir + dep_dir + vctk_pkg, vctk_pkg_path)

if not os.path.exists(vctkdir):
    print("Unpacking vctk...")
    untar_cmd = ["tar", "xzf", vctk_pkg_path]
    pe(untar_cmd)

os.chdir(base_install_dir)
speech_tools_pkg = "speech_tools-2.4-release.tar.gz"
speech_tools_pkg_path = base_synthesis_dir + speech_tools_pkg
if not os.path.exists(speech_tools_pkg_path):
    os.symlink(full_dep_dir + speech_tools_pkg, speech_tools_pkg_path)

os.chdir(base_synthesis_dir)
if not os.path.exists(estdir):
    print("Unpacking speech_tools...")
    untar_cmd = ["tar", "xzf", speech_tools_pkg_path]
    pe(untar_cmd)

# rough check if speech_tools is built or not, if not build it
if not os.path.exists(estdir + "bin/siod"):
    # apparently we expect exist status 2???
    os.chdir(estdir)
    configure_cmd = ["./configure"]
    pe(configure_cmd)
    make_cmd = ["make", "-j", "4"]
    pe(make_cmd)

# Install festival
os.chdir(base_synthesis_dir)
festival_pkg = "festival-2.4-release.tar.gz"
festival_pkg_path = base_synthesis_dir + festival_pkg
if not os.path.exists(festival_pkg_path):
    os.symlink(full_dep_dir + festival_pkg, festival_pkg_path)

if not os.path.exists(festdir):
    untar_cmd = ["tar", "xzf", festival_pkg_path]
    pe(untar_cmd)

if not os.path.exists(festdir + "bin/festival"):
    os.chdir(festdir)
    configure_cmd = ["./configure"]
    pe(configure_cmd)
    make_cmd = ["make"]
    pe(make_cmd)

# Install festival addons
# festlex_CMU
# festlex_OALD
# festlex_POSLEX
# festvox_cmu_us_slt_cg.tar.gz
cmu_lex_pkg = "festlex_CMU.tar.gz"
cmu_lex_pkg_path = base_synthesis_dir + cmu_lex_pkg
if not os.path.exists(cmu_lex_pkg_path):
    os.symlink(full_dep_dir + cmu_lex_pkg, cmu_lex_pkg_path)

oald_pkg = "festlex_OALD.tar.gz"
oald_pkg_path = base_synthesis_dir + oald_pkg
if not os.path.exists(oald_pkg_path):
    os.symlink(full_dep_dir + oald_pkg, oald_pkg_path)

poslex_pkg = "festlex_POSLEX.tar.gz"
poslex_pkg_path = base_synthesis_dir + poslex_pkg
if not os.path.exists(poslex_pkg_path):
    os.symlink(full_dep_dir + poslex_pkg, poslex_pkg_path)

slt_cg_pkg = "festvox_cmu_us_slt_cg.tar.gz"
slt_cg_pkg_path = base_synthesis_dir + slt_cg_pkg
if not os.path.exists(slt_cg_pkg_path):
    os.symlink(full_dep_dir + slt_cg_pkg, slt_cg_pkg_path)

os.chdir(base_synthesis_dir)
if not os.path.exists(festdir + "lib/voices"):
    # if no voice dir install all the lex stuff...
    untar_cmd = ["tar", "xzf", slt_cg_pkg_path]
    pe(untar_cmd)
    untar_cmd = ["tar", "xzf", poslex_pkg_path]
    pe(untar_cmd)
    untar_cmd = ["tar", "xzf", oald_pkg_path]
    pe(untar_cmd)
    untar_cmd = ["tar", "xzf", cmu_lex_pkg_path]
    pe(untar_cmd)

# Install festvox
os.chdir(base_synthesis_dir)
festvox_pkg = "festvox-2.7.0-release.tar.gz"
festvox_pkg_path = base_synthesis_dir + festvox_pkg
if not os.path.exists(festvox_pkg_path):
    os.symlink(full_dep_dir + festvox_pkg, festvox_pkg_path)

if not os.path.exists(festvoxdir):
    untar_cmd = ["tar", "xzf", festvox_pkg_path]
    pe(untar_cmd)

# build it
if not os.path.exists(festvoxdir + "src/ehmm/bin/ehmm"):
    os.chdir(festvoxdir)
    configure_cmd = ["./configure"]
    pe(configure_cmd)
    make_cmd = ["make"]
    pe(make_cmd)

# Install htk
# patch for HTS
os.chdir(base_synthesis_dir)
htk_pkg = "HTK-3.4.1.tar.gz"
htk_pkg_path = base_synthesis_dir + htk_pkg
if not os.path.exists(htk_pkg_path):
    os.symlink(full_dep_dir + htk_pkg, htk_pkg_path)

if not os.path.exists(htkdir):
    untar_cmd = ["tar", "xzf", htk_pkg_path]
    pe(untar_cmd)

if not os.path.exists(htkdir + "HTKTools/HSGen"):
    # HTS patchfile
    os.chdir(base_synthesis_dir)
    hts_patch_pkg = "HTS-2.3_for_HTK-3.4.1.tar.bz2"
    patch_dir = "hts_patch/"
    hts_patch_dir = base_synthesis_dir + patch_dir
    hts_patch_path = hts_patch_dir + hts_patch_pkg
    if not os.path.exists(hts_patch_pkg):
        if not os.path.exists(hts_patch_dir):
            os.mkdir(hts_patch_dir)
        if not os.path.exists(hts_patch_path):
            os.symlink(full_dep_dir + hts_patch_pkg, hts_patch_path)

    full_patch_path = hts_patch_dir + "HTS-2.3_for_HTK-3.4.1.patch"
    os.chdir(hts_patch_dir)
    untar_cmd = ["tar", "xjf", hts_patch_path]
    pe(untar_cmd)
    os.chdir(htkdir)
    try:
        pe("patch -p1 -d . -f < %s" % full_patch_path, shell=True)
    except subprocess.CalledProcessError:
        # we expect the patch to partially fail :/
        pass

    os.chdir(htkdir)
    pe(["./configure", "--disable-hlmtools", "--disable-hslab"])
    pe(["make"])

os.chdir(base_synthesis_dir)
sptk_pkg = "SPTK-3.9.tar.gz"
sptk_subdir = base_synthesis_dir + "sptk/"
sptk_pkg_path = sptk_subdir + sptk_pkg
if not os.path.exists(sptk_subdir):
    os.mkdir(sptk_subdir)

if not os.path.exists(sptk_pkg_path):
    os.symlink(full_dep_dir + sptk_pkg, sptk_pkg_path)

# Install sptk
if not os.path.exists(sptkdir):
    os.chdir(sptk_subdir)
    untar_cmd = ["tar", "xzf", "SPTK-3.9.tar.gz"]
    pe(untar_cmd)
    os.chdir("SPTK-3.9")
    os.mkdir("out")
    pe(["./configure", "--prefix=%s" % sptk_subdir + "SPTK-3.9/out"])
    pe(["make"])
    os.chdir(sptk_subdir + "SPTK-3.9")
    pe(["make install"], shell=True)
    os.chdir(base_synthesis_dir)
    os.mkdir("SPTK-3.9")
    copytree("sptk/SPTK-3.9/out", "SPTK-3.9")

os.chdir(base_synthesis_dir)
hts_engine_pkg = "hts_engine_API-1.10.tar.gz"
hts_engine_pkg_path = base_synthesis_dir + hts_engine_pkg
if not os.path.exists(hts_engine_pkg_path):
    os.symlink(full_dep_dir + hts_engine_pkg, hts_engine_pkg_path)

if not os.path.exists(htsenginedir):
    untar_cmd = ["tar", "xzf", hts_engine_pkg_path]
    pe(untar_cmd)

# Install hts engine
os.chdir(htsenginedir)
if not os.path.exists(htsenginedir + "bin/hts_engine"):
    configure_cmd = ["./configure"]
    pe(configure_cmd)
    make_cmd = ["make"]
    pe(make_cmd)

os.chdir(base_synthesis_dir)
hts_demo_pkg = "HTS-demo_CMU-ARCTIC-SLT.tar.bz2"
hts_demo_pkg_path = base_synthesis_dir + hts_demo_pkg
if not os.path.exists(hts_demo_pkg_path):
    os.symlink(full_dep_dir + hts_demo_pkg, hts_demo_pkg_path)

# Unpack HTS demo
if not os.path.exists(htsdemodir):
    untar_cmd = ["tar", "xjf", hts_demo_pkg_path]
    pe(untar_cmd)

if not os.path.exists(htsdemodir + "data/lf0/cmu_us_arctic_slt_a0001.lf0"):
    os.chdir(htsdemodir)
    configure_cmd = ["./configure"]
    configure_cmd += ["--with-fest-search-path=%s" % (festdir + "examples")]
    configure_cmd += ["--with-sptk-search-path=%s" % (sptkdir + "bin")]
    configure_cmd += ["--with-hts-search-path=%s" % (htkdir + "HTKTools")]
    configure_cmd += ["--with-hts-engine-search-path=%s" % (htsenginedir + "bin")]
    pe(configure_cmd)

print("Typing 'make' in %s will run a speech sythesis demo, but it takes a long time" % htsdemodir)
print("Also dumping a helper source script to %stts_env.sh" % base_synthesis_dir)
# http://www.nguyenquyhy.com/2014/07/create-full-context-labels-for-hts/
lns = ["export ESTDIR=%s\n" % estdir]
lns.append("export FESTDIR=%s\n" % festdir)
lns.append("export FESTVOXDIR=%s\n" % festvoxdir)
lns.append("export VCTKDIR=%s\n" % vctkdir)
lns.append("export HTKDIR=%s\n" % htkdir)
lns.append("export SPTKDIR=%s\n" % sptkdir)
lns.append("export HTSENGINEDIR=%s\n" % htsenginedir)
lns.append("export HTSDEMODIR=%s\n" % htsdemodir)
lns.append("export HTSPATCHDIR=%s\n" % htspatchdir)
lns.append("export MERLINDIR=%s\n" % merlindir)

os.chdir(base_synthesis_dir)
with open("tts_env.sh", "w") as f:
    f.writelines(lns)
