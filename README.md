# cs498 project

We have two scripts to produce predictions, one based on an ensemble of classical machine learning algorithms (model_predictions_ensemble.py) and one based on a deep neural network (insert_here.py).

The pickle, sys, numpy, and pandas packages are necessary to run model_predictions_ensemble.py.
To obtain the predictions simply execute: 

python model_predictions_ensemble.py <path-to-dataset>

Results will be written to a file named "ensemble_output.txt"

Note: This is our main classifier for the project.

We have also exprerimented with a Bidirectional recurrent neural network architecture to demostrate its ability on such text classification tasks where the sequence of input matters.
The code for this is in the RNN folder.
The training along with test results are shown in the Topic Prediction using Bidir LSTM-Training.ipynb file.

To run inference - python rnn_inference.py <path-to-dataset>

Note: This is an experimental classifier to try an alternative approach & not the main classifier used for final predictions for this project. Thus it performance not to be taken into account. 
