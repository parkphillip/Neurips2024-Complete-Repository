# Facial-Emotion-Recognition


<!-- ABOUT THE PROJECT -->
## About The Project

A tensorflow/keras implementation of dual-model approach to emotion recognition. Was created for the Neurips High School Project Track. 
Paper titled: Towards Inclusive Emotion Recognition: Dual-Model
Deep Learning for Social and Emotional Support

### Built With
* Keras
* Tensorflow
  
<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* python >= 3.7.9
* keras >= 2.4.3
* tensorflow >= 2.3.1
* opencv >= 4.4
* sklearn >= 0.23
* numpy >= 1.18.5
* pandas >= 1.1.2
* matplotlib >= 3.3.1

## Dataset
The dataset used for the speech aspect of this work is the [Ryerson Audio-Visual Database of Emotional Speech and Song](https://smartlaboratory.org/ravdess/) (RAVDESS). This dataset contains audio and visual recordings of 12 male and 12 female actors pronouncing English sentences with eight different emotional expressions. For this task, we utilize only speech samples from the database with the following eight different emotion classes: sad, happy, angry, calm, fearful, surprised, neutral and disgust.

The dataset used for the facial recognition aspect of the work is the [FER2013] Dataset. The dataset contains 48×48 pixel grayscale images with 7 different emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The dataset contains 28709 examples in the training set, 3589 examples in the public testing set, and 3589 examples in the private test set. can be found here: https://www.kaggle.com/datasets/msambare/fer2013/code


## Acknowledgements
I would like to express my gratitude to the Irvine Center for Autism \& Neurodevelopmental Disorders. Their work inspires me every day to continue my research in the field of emotion recognition and support for neurodevelopmental conditions. Additionally, I would like to extend my sincere thanks to Samuel Kim, Ph.D., for providing the necessary computational resources that were critical for training my model. His support has been invaluable to this project.

Furthermore, I am deeply appreciative of the opportunities provided by the NeurIPS Committee and reviewers. This platform has enabled aspiring high school researchers like myself to publish our research, an opportunity that would have been otherwise unattainable.
