folder_path="$(pwd)/dataset"
container_name="yolov4-midgard"
configfile=$folder_path/train_config.json
interactive_options='--gpus all -it --runtime=nvidia'
run="$(date '+%Y%m%d_%H:%M:%S')"
ports=''
enable_training=1


COMMAND="/bin/bash"

while test $# -gt 0
do
    case "$1" in
        --run) COMMAND="python3 src/main.py"
            ;;
        --inference)
			set="$2"
			input="$3"
			training_set="/training/custom_training/$set"
			COMMAND="/training/darknet/darknet detector test $training_set/config/obj.data $training_set/config/yolo4.cfg $training_set/weights/yolo4_best.weights $input"
			shift
			shift
            ;;
		--test-darknet)
			COMMAND="ls /training/darknet/darknet"
			interactive_options=''
			;;
		--inference-only)
			enable_training=0
			run="$2"
			shift
			;;
    esac
    shift
done


if [ -f $configfile ]; then
	custom_api=`jq .training.custom_api.enable $configfile`
	if [ "$custom_api" = "true" ]; then
		custom_api_port=`jq .training.custom_api.port $configfile`
		ports="$ports -p $custom_api_port:$custom_api_port"
	fi
	tensorboard=`jq .training.tensorboard.enable $configfile`
	if [ "$tensorboard" = "true" ]; then
		tensorboard_port=`jq .training.tensorboard.port $configfile`
		ports="$ports -p $tensorboard_port:$tensorboard_port"
	fi
	web_ui=`jq .training.web_ui.enable $configfile`
	if [ "$web_ui" = "true" ]; then
		web_ui_port=`jq .training.web_ui.port $configfile`
		ports="$ports -p $web_ui_port:$web_ui_port"
	fi

	docker run  --rm \
				$interactive_options \
				-e TRAIN_NAME=$container_name \
				-e TRAIN_START_TIME=$run \
				-e ENABLE_TRAINING=$enable_training \
				-v $(pwd)/src:/training/src \
				-v $(pwd)/config:/training/config \
				-v $(pwd)/runs:/training/runs \
				-v $folder_path:/training/assets \
				-v $(pwd)/trainings:/training/custom_training \
				-v /etc/timezone:/etc/timezone:ro \
				-v /etc/localtime:/etc/localtime:ro \
				--name $container_name \
				$ports \
				$USER/yolov4:latest \
				$COMMAND
else
	echo "Error: Configuration file not found in the provided path"
fi
