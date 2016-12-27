for f in *.ipynb ; do echo $f; aspell list < "$f" | sort | uniq -c ; done | less
2to3 -f imports folder/path | patch -p0