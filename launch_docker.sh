folder_path="$(pwd)/dataset"
container_name="yolov4-midgard"
configfile=$folder_path/train_config.json
ports=''

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

	docker run  --gpus all --rm --runtime=nvidia -it \
				-e TRAIN_NAME=$container_name \
				-e TRAIN_START_TIME="$(date '+%Y%m%d_%H:%M:%S')" $ports \
				-v $folder_path:/training/assets \
				-v $(pwd)/trainings:/training/custom_training \
				-v /etc/timezone:/etc/timezone:ro \
				-v /etc/localtime:/etc/localtime:ro \
				--name $container_name \
				$USER/yolov4:latest \
				python3 src/main.py
else
	echo "Error: Configuration file not found in the provided path"
fi
