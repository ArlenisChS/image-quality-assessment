sudo ./evaluate  \
--docker-image nima-cpu \
--base-model-name MobileNet \
--weights-file $(pwd)/models/MobileNet/weights_mobilenet_06_0.237.hdf5 \
--n-classes 5 \
--image-json $(pwd)/data/ImageQuality/document_quality_test.json \
--image-dir /home/arlenis/datasets/ImageQuality/