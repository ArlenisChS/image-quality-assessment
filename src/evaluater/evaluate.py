import argparse
import numpy as np
from tensorflow.python.keras.backend import dtype
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator
from utils.utils import calc_mean_score, save_json, load_json
from sklearn.metrics import confusion_matrix, accuracy_score
from os.path import join

def get_labels(samples, ):
    return [sample['label'].index(1)+1 for sample in samples], \
        [sample['mean_score_prediction'] for sample in samples]

def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)

def main(base_model_name, weights_file, n_classes, image_dir, image_json, results_dir, img_format='jpg'):
    # load samples
    samples = load_json(image_json)

    # build model and load weights
    nima = Nima(base_model_name, n_classes=n_classes, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, n_classes, nima.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions = predict(nima.nima_model, data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i], n_classes=n_classes)

    # get real and estimated labels
    x_labels, y_labels = get_labels(samples)
    y_labels = np.array(np.ceil(y_labels), dtype=int)
    
    # estimate metrics
    results = {
        "samples": samples,
        "accuracy score": accuracy_score(x_labels, y_labels),
        "confusion matrix": confusion_matrix(x_labels, y_labels).tolist()
    }

    print(results['samples'])
    print(results['accuracy score'])
    print(results['confusion matrix'])

    if results_dir is not None:
        save_json(results['samples'], join(results_dir, 'samples.json'))
        save_json(results['accuracy score'], join(results_dir, 'accuracy score.json'))
        save_json(results['confusion matrix'], join(results_dir, 'confusion matrix.json'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-nc', '--n-classes', help='classification classes', required=False, type=int, default=10)
    parser.add_argument('-ij', '--image-json', help='json with tagged images', required=True)
    parser.add_argument('-id', '--image-dir', help='image directory', required=True)
    parser.add_argument('-rf', '--results-dir', help='results directory', required=False, default=None)

    args = parser.parse_args()
    main(**args.__dict__)
