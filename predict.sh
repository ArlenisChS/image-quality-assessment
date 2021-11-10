sudo ./predict  \
--docker-image nima-cpu \
--base-model-name MobileNet \
--weights-file $(pwd)/models/MobileNet/weights_mobilenet_technical_0.11.hdf5 \
--n-classes 10 \
--image-source $(pwd)/src/tests/test_images