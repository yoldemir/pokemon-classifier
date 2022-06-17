## Transfer learning example using Keras

This repository contains a simple transfer learning example using Keras that does Pokemon classification. Weights of the pretrained ResNet50 model are frozen, and only the weights of the fully connected layers appended to the end of the network are trained. 

The dataset in this repository is tracked using Git LFS.

To run the code:
- `virtualenv -p python3 my_env`
- `source my_env/bin/activate`
- `pip install -r requirements.txt`
- `python main.py`
