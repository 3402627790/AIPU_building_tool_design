#!/bin/bash
DOC_DIR=/project/ai/scratch01/AIPUBuilder/
echo "generating doc..."
doxygen Doxyfile
echo "generating done!"
echo "coping doc to $DOC_DIR"
cp -r html $DOC_DIR
echo "done!"
