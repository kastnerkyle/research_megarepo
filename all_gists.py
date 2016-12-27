#!/usr/bin/env python
# Clone or update all a user's gists
# curl -ks https://raw.github.com/gist/5466075/gist-backup.py | USER=fedir python
# USER=fedir python gist-backup.py

import json
from six.moves import urllib
from subprocess import call, Popen, PIPE
import os
import math
import shutil

if not os.path.exists("gist_dump"):
    os.mkdir("gist_dump")
os.chdir("gist_dump")

USER = 'kastnerkyle'
EMAIL = "kastnerkyle@gmail.com"

perpage = 30.0
userurl = urllib.request.urlopen('https://api.github.com/users/' + USER).read().decode('utf-8')
public_gists = json.loads(userurl)
gistcount = public_gists['public_gists']
print("Found gists : " + str(gistcount))
pages = int(math.ceil(float(gistcount)/perpage))
print("Found pages : " + str(pages))

f=open('./contents.txt', 'w+')

GIT_COMMIT_FIELDS = ['id', 'author_name', 'author_email', 'date', 'message']
GIT_LOG_FORMAT = ['%H', '%an', '%ae', '%ad', '%s']
GIT_LOG_FORMAT = '%x1f'.join(GIT_LOG_FORMAT) + '%x1e'

lu_file = open("./README.md", "w")

include_pure_forks = False

for page in range(pages):
    pageNumber = str(page + 1)
    print("Processing page number " + pageNumber)
    pageUrl = 'https://api.github.com/users/' + USER  + '/gists?page=' + pageNumber + '&per_page=' + str(int(perpage))
    u = urllib.request.urlopen(pageUrl).read().decode('utf-8')
    gists = json.loads(u)
    startd = os.getcwd()
    for gist in gists:
        gistd = gist['id']
        gistUrl = 'git://gist.github.com/' + gistd + '.git'
        """
        # only 60 API calls an hour... cmon github
        if not include_pure_forks:
            c = urllib.request.urlopen(gist["commits_url"]).read().decode('utf-8')
            commits = json.loads(c)
            # Check for pure forks e.g. ones where origin is not USER and last
            if commits[0]['user']['login'] != USER and commits[-1]['user']['login'] != USER:
                continue
        """
        if os.path.isdir(gistd):
            os.chdir(gistd)
            call(['git', 'pull', gistUrl])
            os.chdir(startd)
        else:
            call(['git', 'clone', gistUrl])

        dumpdir = os.getcwd()
        if include_pure_forks is False:
           os.chdir(gistd)
           p = Popen('git log --format="%s"' % GIT_LOG_FORMAT, shell=True, stdout=PIPE)
           (log, _) = p.communicate()
           log = log.strip('\n\x1e').split("\x1e")
           log = [row.strip().split("\x1f") for row in log]
           log = [dict(zip(GIT_COMMIT_FIELDS, row)) for row in log]
           if log[0]["author_email"] != EMAIL and log[-1]["author_email"] != EMAIL:
               os.chdir(dumpdir)
               shutil.rmtree(gistd)
           else:
               filenames = [f for f in os.listdir(".") if f != ".git"]
               copynames = [gistd for f in filenames]
               if len(filenames) > 1:
                   for i in range(1, len(filenames)):
                       copynames[i] = copynames[i] + "_%i" % i
               os.chdir(dumpdir)
               for f, c in zip(filenames, copynames):
                   orig = f
	           shutil.copy2(gistd + "/" + orig, orig)
                   lu_file.writelines(["%s : https://gist.github.com/kastnerkyle/%s\n\n" % (orig, gistd)])
	       shutil.rmtree(gistd)
        os.chdir(dumpdir)
