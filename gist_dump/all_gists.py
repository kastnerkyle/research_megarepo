#!/usr/bin/env python
# Clone or update all a user's gists
# curl -ks https://raw.github.com/gist/5466075/gist-backup.py | USER=fedir python
# USER=fedir python gist-backup.py

import json
import urllib.request, urllib.parse, urllib.error
from subprocess import call
from urllib.request import urlopen
import os
import math

USER = 'kastnerkyle'
perpage=30.0
userurl = urlopen('https://api.github.com/users/' + USER).readall().decode('utf-8')
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
    u = urlopen(pageUrl).readall().decode('utf-8')
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