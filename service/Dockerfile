FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN apt-get update && apt-get install libturbojpeg

WORKDIR /detector_segmentation

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY . /detector_segmentation/

EXPOSE 8892

CMD make run_app
