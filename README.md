The above scripts were written and tested with a particular file system in place. In the absence of the same file system, we expect the code to fail. For ex, in raspberryPi the web-server script is only read from "/var/www/html/index.html".

The inference code is itself simple and does not depend on other files except the model architecture. The inference code contains model loading and invoking of its tensor weights. Thus, we must know the datatype of the model parameters. Apart from this, the inference code expects a "webpage.txt" file containing the web server code. You can make changes in the webpage.txt file, and the same changes will reflect in the index.html.

While testing the scripts, many dependencies were installed, many of which are no longer required in the current version.

The script "staticLabelingUsingDTW.py" performs DTW scoring for the unlabeled data to generate labels for the "static-dataset".

The script "modelTraining.ipynb" contains the complete training process of the neural network model. This script expects the raw, unlabeled data from the IMU sensor and also the CSV files generated after running "staticLabelingUsingDTW.py". Furthermore, this script contains the code to label the dynamic side-to-side data from the raw, unlabeled data.
