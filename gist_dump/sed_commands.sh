# skip hidden files and sed replace
find . -not -path '*/\.*' -type f -exec sed -i -e 's|from pelican|from ../lib/pelican|' {} +

# replace in every file recursive
find . -type f -exec sed -i -e 's/foo/bar/g' {} +

# replace in all files in current dir 
sed -i -- 's/foo/bar/g' *

# tmp_wav.txt is raw text from dropbox downloads list - pull out wav files, making sure to strip leading whitespace
cat tmp_wav.txt | grep \\.wav | sed -e "s/^[ \t]*//g"