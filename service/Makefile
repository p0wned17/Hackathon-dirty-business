APP_PORT := 8888
DOCKER_TAG := latest
DOCKER_IMAGE := dirty_segmentation_service
NUM_WORKERS := 1

run_app:
	python -m uvicorn app:create_app --host='0.0.0.0' --port=$(APP_PORT) --workers=$(NUM_WORKERS)

locust_test:
	locust -f locustfile.py --host=http://localhost:8888 --csv=results --logfile=locust.log

run_app_docker:
	docker run --ipc=host --network=host --gpus=all --name detector_segment --restart always -d detector_segment:latest

docker_build:
	docker build -f Dockerfile . --force-rm=true -t detector_segment:latest