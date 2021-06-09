import os
import librosa as lb
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras

TRAINED_MODEL = "music_genre_classifier.h5"  # Trained model

'''
Since I have segmented the audio samples into 10 while training we have to pass
the same value of samples for testing too.
So, we are segmenting the audio samples.
'''
SAMPLE_RATE = 22050
DURATION = 30  # The Duration of the Dataset in the Music Genres are of 30 Seconds each.
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
num_segments = 10
hop_length = 512
num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

segment = 1  # Specify the segment which you want to find out

# Specify the songs you want to classify

PATH_1 = "Test\\blues.00091.wav"
PATH_2 = "Test\\country.00021.wav"
PATH_3 = "Test\\disco.00010.wav"


def preprocess(file_path, n_mfcc=13, n_fft=2048):
    # Load the audio file
    signal, sr = lb.load(file_path)

    start_sample = num_samples_per_segment * segment
    finish_sample = start_sample + num_samples_per_segment

    mfcc = lb.feature.mfcc(signal[start_sample:finish_sample],
                           sr=SAMPLE_RATE,
                           n_fft=n_fft,
                           n_mfcc=n_mfcc,
                           hop_length=hop_length)
    mfcc = mfcc.T
    return mfcc


class _Genre_classification_Service:
    model = None
    _mappings = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ]

    _instance = None

    def predict(self, file_path):
        # Extract the MFCCs
        MFCCs = preprocess(file_path)

        # Convert 2D MFCCs to 4D
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # Make Predictions
        prediction = self.model.predict(MFCCs)
        predicted_index = np.argmax(prediction)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


def music_genre_classifier():
    # Ensure that we only have one instance
    if _Genre_classification_Service._instance is None:
        _Genre_classification_Service._instance = _Genre_classification_Service()
        _Genre_classification_Service.model = keras.models.load_model(TRAINED_MODEL)
    return _Genre_classification_Service._instance


if __name__ == "__main__":
    mgc = music_genre_classifier()

    Genre1 = mgc.predict(PATH_1)
    Genre2 = mgc.predict(PATH_2)
    Genre3 = mgc.predict(PATH_3)

    print(f"Predicted Keyword: {Genre1},{Genre2}, {Genre3}")
