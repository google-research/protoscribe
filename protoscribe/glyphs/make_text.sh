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

# Usage, e.g.:
#
# protoscribe/glyphs/make_text.sh \
#  -t X,X,I,I,CLOTH,SHIRT \
#  -p /var/tmp/22_shirts.png \
#  -d  # If you want to display the result with Eye of GNOME (eog) viewer.

while getopts 'b:t:p:s:S:d' OPTION
do
  case "${OPTION}" in
    b)
      BCKGRND="${OPTARG}"
      ;;
    t)
      TERMS="${OPTARG}"
      ;;
    p)
      PNG="${OPTARG}"
      ;;
    s)
      SVG="${OPTARG}"
      ;;
    S)
      SVG_FOR_STROKES="${OPTARG}"
      ;;
    d)
      DISPLAY_PNG=true
      ;;
  esac
done
if [ "${BCKGRND}" == "" ]
then
  BCKGRND="ivory"
fi

SRC_DIR=protoscribe
"python ${SRC_DIR}/glyphs/make_text"_main.py \
  --random_resize \
  --random_pad \
  --random_rotate \
  --extra_pad=50 \
  --concepts="${TERMS}" \
  --background_color="${BCKGRND}" \
  --svg_output="${SVG}" \
  --svg_for_strokes_output="${SVG_FOR_STROKES}" \
  --output="${PNG}"
if [ "${DISPLAY_PNG}" == "true" ]
then
  eog "${PNG}"
fi
