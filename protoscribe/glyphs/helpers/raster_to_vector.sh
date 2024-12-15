# Copyright 2024 The Protoscribe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
#
# Converts input image in (PNG/JPG/etc.) format to SVG.
#
# Please note: You can *also* run this utility *directly* on the input SVGs.
#
# Dependencies:
# -------------
#   sudo apt-get install imagemagick potrace

# From https://stackoverflow.com/questions/56696496/how-to-convert-jpg-or-png-image-to-svg-and-save-it:
#
# Following converts PNG file to PGM format, removes image transparency, outputs
# the result image to the standard input of `mkbitmap` that transforms the input
# with highpass filtering and thresholding into a suitable for the `potrace`
# program format, that finally generates SVG file.
#
# You can play around with highpass filtering (-f) and thresholding (-t) values
# until you have the final look that you want.

while getopts 'i:o:' OPTION ; do
  case "$OPTION" in
    i)
      INPUT_FILE="$OPTARG"
      ;;
    o)
      OUTPUT_FILE="$OPTARG"
      ;;
    *)
      echo "Usage: $(basename $0) -i <INPUT> -o <OUTPUT>" >&2
      exit 1
      ;;
  esac
done

if [ ! -f "${INPUT_FILE}" ] ; then
  echo "Input file does not exist!"
  exit 1
fi
if [ -z "${OUTPUT_FILE}" ] ; then
  echo "Output file not specified!"
  exit 1
fi

# This is not trivial as the background is not necessarily white.
TMPFILE_PGM=$(mktemp --suffix .pgm /tmp/raster2vector.XXXXXX)
convert -verbose "${INPUT_FILE}" \
  -fuzz 15% -flatten -alpha remove -background white ${TMPFILE_PGM} \
  || { echo "convert failed!" >&2; exit 1; }
TMPFILE_BMP=$(mktemp --suffix .bmp /tmp/raster2vector.XXXXXX)
mkbitmap -f 32 -t 0.4 ${TMPFILE_PGM} -o ${TMPFILE_BMP} \
  || { echo "mkbitmap failed!" >&2; exit 1; }
potrace --svg ${TMPFILE_BMP} -o "${OUTPUT_FILE}" \
  || { echo "potrace failed!" >&2; exit 1; }
rm -f ${TMPFILE_PGM} ${TMPFILE_BMP}
