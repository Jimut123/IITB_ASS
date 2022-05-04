# pip install --upgrade --no-cache-dir gdown
# gdown 1GgA1BQ0fthGD1PuAwcETvHdSigpyFxK7

unzip -qq Kolkata_020.zip

# remove all the roads images
rm -rf Kolkata_020/*.png

du -hs Kolkata_020

mkdir Kolkata_020/train
mkdir Kolkata_020/test

