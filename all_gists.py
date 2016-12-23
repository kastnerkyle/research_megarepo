#!/usr/bin/env python
# Clone or update all a user's gists
# curl -ks https://raw.github.com/gist/5466075/gist-backup.py | USER=fedir python
# USER=fedir python gist-backup.py

import json
from six.moves import urllib
from subprocess import call
import os
import math
if not os.path.exists("gist_dump"):
    os.mkdir("gist_dump")
os.chdir("gist_dump")

USER = 'kastnerkyle'
perpage=30.0
userurl = urllib.request.urlopen('https://api.github.com/users/' + USER).read().decode('utf-8')
public_gists = json.loads(userurl)
gistcount = public_gists['public_gists']
print("Found gists : " + str(gistcount))
pages = int(math.ceil(float(gistcount)/perpage))
print("Found pages : " + str(pages))

f=open('./contents.txt', 'w+')

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
        if os.path.isdir(gistd):
            os.chdir(gistd)
            call(['git', 'pull', gistUrl])
            os.chdir(startd)
        else:
            call(['git', 'clone', gistUrl])
