#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from PIL import Image
from PIL.ExifTags import TAGS
import glob

def get_exif(img, field='DateTimeOriginal'):
    for k, v in img._getexif().items():
        if TAGS.get(k) == field:
            return v
    return None

if len(sys.argv) == 1:
    print( "usage:")
    sys.exit(-1)

if sys.argv[1] == '-a':
    fnames = glob.glob("*.JPG")
else:
    fnames = argv[1:]
print( fnames )
    
for fname in fnames:
    newname = ""
    with Image.open(fname) as jpg_image:
        val = get_exif(jpg_image)
        newname = val.replace(':','').replace( " ","_")+".jpg"
        newname = newname[:4]+"_"+newname[4:]

    if( newname ):
        print( fname, '->', newname)
        os.rename(fname,newname)
    else:
        print("no sacchi file " + fname )
