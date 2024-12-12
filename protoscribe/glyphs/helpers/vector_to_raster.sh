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
# Converts SVGs to PNGs.
#
# Dependencies:
# -------------
#   sudo apt-get install librsvg2-bin

WIDTH=256px
HEIGHT=256px
BACKGROUND=
while getopts 'i:o:h:w:b' OPTION ; do
  case "$OPTION" in
    i)
      INPUT_FILE="$OPTARG"
      ;;
    o)
      OUTPUT_FILE="$OPTARG"
      ;;
    w)
      WIDTH="$OPTARG"
      ;;
    h)
      HEIGHT="$OPTARG"
      ;;
    b)
      BACKGROUND="-b white"
      ;;
    ?)
      echo "Usage: $(basename \$0) -i INPUT -o OUTPUT" >&2
      exit 1
      ;;
  esac
done

rsvg-convert \
  --width=${WIDTH} --height=${HEIGHT} \
  --keep-aspect-ratio ${BACKGROUND} ${INPUT_FILE} > ${OUTPUT_FILE} \
  || { echo "rsvg-convert failed!" >&2; exit 1; }
