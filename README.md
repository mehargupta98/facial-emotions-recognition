
# Facial Emotion Recognition using CNNs

Facial Emotion Recognition (FER) is the technology that analyses facial expressions from both static images and videos in order to reveal information on one’s emotional state. The complexity of facial expressions, the potential use of the technology in any context, and the involvement of new technologies such as artificial intelligence raise significant privacy risks.

In this project, a model is created using CNN for recognising the facial emotion from muxspace dataset consisting of static images available on GitHub.

## 1. Load Dataset

We will be using the dataset https://github.com/muxspace/facial_expressions, it has images of faces along with their emotion labels.

This dataset contains 8 Emotions :- Anger, Surprise, Disgust, Fear, Happiness, Sadness, Contempt, Neutral

## 2. Data Preprocessing

We use a dictionary in which the key is the emotion associated with the image, while value is a list of images for that particular emotion.

We then use the above dictionary to place our images in a directory structure which will be picked by ImageDataGenerator, and split the data into train and test.

## 3. ImageDataGenerator
We use flow_from_directory method of Kera’s ImageDataGenerator. This method is useful when the images are sorted and placed in their respective class/label folders. This method will identify classes automatically from the folder name (Will do one-hot encoding).

## 4. Model Creation

- ### Feature Extraction

This architecture of the model involves stacking convolutional layers with small 3×3 filters followed by a max pooling layer. Together, these layers form a block, and these blocks are repeated where the number of filters in each block is increased with the depth of the network as 32, 64 and 128. 

Each layer uses the ReLU activation function, which is generally a best practice.

- ### Output

Now, the feature maps output from the feature extraction part of the model must be flattened. We then interpret them with one or more fully connected layers, and then output a prediction. The output layer has 8 nodes for the 8 classes and use the softmax activation function.

We ran this model for total 10 epochs but it can be experimented further.
