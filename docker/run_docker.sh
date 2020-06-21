
export DOCKER_CONTAINER_NAME=docker_container
if [ ! "$(docker ps -q -f name=${DOCKER_CONTAINER_NAME})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${DOCKER_CONTAINER_NAME})" ]; then
        echo "Cleaning up existing container named ${DOCKER_CONTAINER_NAME}"
        # cleanup
        docker rm $DOCKER_CONTAINER_NAME
    fi
    # run your container 
    docker run -it --rm \
      --name "$DOCKER_CONTAINER_NAME" \
      --env="NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}" \
      --env="NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics" \
     -p10000:10000 \
      --runtime=nvidia \
      --gpus all \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      --volume="/home/guido/code/online_lstm_episodic_memory:/home/green/online_lstm_episodic_memory/:rw"  \
      guidoski/greenhouse:latest bash
fi
