# A learning without forgetting approach to incorporate artifact knowledge in polyp localization tasks

This repository contains the code used in our work "A learning without forgetting approach to 
incorporate artifact knowledge in polyp localization tasks" [[**arXiv**]](https://arxiv.org/abs/2002.02883).
The code is based on Keras with the Tensorflow backend. 

Currently, only the base artifact detector is available. We will be adding the remaining models for polyp localization, 
and multi-task polyp-artifact detection soon. 

## Overview
Colorectal polyps are abnormalities in the colon tissue that can develop into colorectal cancer. 
The survival rate for patients is higher when the disease is detected at an early stage and polyps 
can be removed before they develop into malignant tumors. Deep learning methods have become the state 
of art in automatic polyp detection. However, the performance of current models heavily relies on the 
size and quality of the training datasets. Endoscopic video sequences tend to be corrupted by different
artifacts affecting visibility and hence, the detection rates. In this work, we analyze the effects that 
artifacts have in the polyp localization problem. For this, we evaluate the RetinaNet architecture, 
originally defined for object localization. We also define a model inspired by the learning without 
forgetting framework, which allows us to employ artifact detection knowledge in the polyp localization 
problem. Finally, we perform several experiments to analyze the influence of the artifacts in the 
performance of these models. To our best knowledge, this is the first extensive analysis of the 
influence of artifact in polyp localization and the first work incorporating learning without 
forgetting ideas for simultaneous artifact and polyp localization tasks.

## Usage and Requirements

The code was tested with python3 and CUDA 9.2. To run the code, clone the repository

```bash
$ git clone https://github.com/rodsom22/lwf_polyps.git
```
and install the required packages

```bash
$ cd lwf_polyps
$ pip install -r requirements.txt
```
## Data Preparation

To run the code, place the image data in the folder `data` together with the following files: 
* A class map: a csv file containing a mapping between class names and class ids. See `data/artifacts.csv` for an example
  with artifact classes. 
  
* Bounding box annotations: A CSV file with bounding box annotations for the images. This is used for validations 
purposes. The format of the CSV is `img_name, x1, y1, x2, y2, class_name`, where `x1, y1` and `x2, y2` indicate the 
coordinates of the top-left and bottom-right coordinates of the bounding box, respectively. `class_name` indicate the 
bounding box's corresponding class, according to the class names - class ids map provided.  If a particular image does 
not have an annotation, an empty row with the form `img_name,,,,,` can be provided.

After the inference process, the model will output a CSV file with the format 
`image_path, x1, y1, x2, y2, class_name, prediction_score` containing the predicted bounding boxes. 
The folder `data/results/` contains an example for the output of the artifact detector. 

## Base Artifact Detector

Download the [model](https://campowncloud.in.tum.de/index.php/s/IhKEDFAvS5GK1ru) for artifact detection and place it into the `models` folder. 
The, run the detection example code:

```bash
$ python artifact_detector.py
```

## Polyp Localization

The process to run the polyp localization models is similar to the described for artifacts. We will be 
adding the models and examples soon. 

## Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/2002.02883):
```
@misc{soberanismukul2020learning,
    title={A learning without forgetting approach to incorporate artifact knowledge in polyp localization tasks},
    author={Roger D. Soberanis-Mukul and Maxime Kayser and Anna-Maria Zvereva and Peter Klare and Nassir Navab and Shadi Albarqouni},
    year={2020},
    eprint={2002.02883},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Acknowledgement
Our work uses the RetinaNet implementation by [@Fizyr](https://github.com/fizyr/keras-retinanet). 
