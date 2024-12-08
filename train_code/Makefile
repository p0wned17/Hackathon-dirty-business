.PHONY: *
DOCKER_TAG := latest
DOCKER_IMAGE := hackathon_train_env
CONTAINER_NAME := hackathon_train_env
PWD = $(shell pwd)

docker_build:
	docker build -f ./docker/Dockerfile . -t $(DOCKER_IMAGE):$(DOCKER_TAG)

docker_run:
	@echo "Starting docker container"
	docker run -it --rm \
		--ipc=host \
		--network=host \
		--gpus=all \
		-v $(PWD):/workspace/project \
		--name "$(CONTAINER_NAME)" \
		"$(DOCKER_IMAGE)"

download_dataset:
	python yandex_downloader.py -l https://disk.yandex.ru/d/6SZgY9PQm7xP2A
	unzip '2 задача.zip'
	unzip '2 задача/train_dataset.zip' 
	rm -rf '2 задача'
	rm -rf cv_synt_dataset
	rm -rf baseline
	python convert_dataset.py
	rm -rf cv_open_dataset

train:
	python src/train_yolo/train_yolo.py

inference:
	PYTHONPATH=. python inference/inference.py
