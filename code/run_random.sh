#!/bin/bash
DIRNAME=`dirname $0`
usage ()
{
  echo "This script runs in parallel multiple experiments. FOLDER is generated by $DIRNAME/config.py script."
  echo "Usage : $0 FOLDER"
  exit
}
if [[ $# -ne 1 ]] ; then
    usage
    exit 1
fi
find $1 -name *json | parallel -j 12 python $DIRNAME/random_model.py {} ">"{//}/command_random.log 2">&"1