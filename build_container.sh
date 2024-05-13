#!/bin/bash

# For building Tensorflow container sandboxes
# TODO make Tensorflow or Pytorch options
CURRENT_DIR=$( pwd )
SCRIPT_DIR=$( dirname -- $0 )
BASE_DIR=${SCRIPT_DIR}/..
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${BASE_DIR}"
IMAGE_DIR=${CURRENT_DIR}/images/
echo IHOME: $IHOME

Help()
{
    echo "Options:"
    echo "	-n: Required. Name for the image"
    echo "	-d: Path to Singularity definition file. Builds an image from specified definition"
    echo "	-t: Optional tag for Singularity definition file. Active only for -d option"
    echo ""
    echo "Environmental variables are specified in a file ../config/improve.env"
}

while getopts hd:n:t: flag
do
    case "${flag}" in
	h) Help
	   exit;;
	n) NAME=${OPTARG};;
	d) DEFINITION_FILE=${OPTARG};;
	t) TAG=${OPTARG};;
    esac
done

if [[ -z ${NAME} ]] ; then  
    echo "Name of the container is not set. -n option is required" 
    exit -1
elif [[ -z $DEFINITION_FILE ]] ; then
    echo "Definition file is not set. -d option is required"
    exit -1
else
	# only works if DEFINITION_FILE is relative path - add check here
    DEFINITION_FILE=${CURRENT_DIR}/${DEFINITION_FILE}
    echo Definition file: ${DEFINITION_FILE}
    echo "Name: ${NAME}"
fi

DATE=$(date +%Y%m%d)


# singularity version 3.9.4
if [[ -z "$TAG" ]] ; then
	TAG="0.0.1"
fi

IMAGE="$NAME:$TAG"
echo "building image: $IMAGE"
singularity build --fakeroot           \
	$IMAGE_DIR/$IMAGE-${DATE}.sif         \
	$DEFINITION_FILE

#echo "building sandbox from image $IIL/${IMAGE}-${DATE}.sif"
#echo "building sandbox at ${ISL}"

