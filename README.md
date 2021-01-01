# Cross-Modal-Conversion-Between-Voice-and-Face

As humans we can infer a person’s looks from the way they speak, there
is a reason to believe that there exists a co-relation between a person’s
voice and face. This project tries to capture this co-relation and has
the following two goals: Reconstructing a facial image that matches the
audio and generating a voice given a small audio recording and a facial
image. To capture this correlation, our model consists of an utterance
encoder/decoder ,voice encoder, image encoder/decoder and classifiers for
voice and face.


## Problem statement
Our problem statement for this project can be broadly stated as:
1. Generate voice that matches input face image
2. Generate face image that matches voice

## Architecture
![architecture](https://user-images.githubusercontent.com/46570073/103434957-700e4500-4bd6-11eb-8d04-23bdc936cd7a.jpg)

## Neural Net Model
![NetworkArch1](https://user-images.githubusercontent.com/46570073/103434971-8f0cd700-4bd6-11eb-8a2c-9ce9beb85670.jpg)
![NetworkArch2](https://user-images.githubusercontent.com/46570073/103434973-93d18b00-4bd6-11eb-9c20-2556f8548ef9.jpg)

## Instrauctions to run
Get CelebA data from below link and palce it in data folder:
https://drive.google.com/file/d/127vi0kFNvYCPj7qJQEFvsFlxdgsqO4iX/view?usp=sharing

Get preptrained model if required from below link and palce in main folder:
https://drive.google.com/file/d/1cfG5chtcdbxEjFDPL9uEW6Iwf-Yf1mRr/view?usp=sharing

Use train jupyter notebook to train
and Use test jupyter notebook to get conversion results
