## URECA Project - 2D Pose Activity Recognition

This is my first attempt to "productionize" ML models, by deploying them to the browser for inferencing. I would admit that this project's design is really hacky, everything is barely coming together.

Interesting stuff to take a look:
1. ``` app/views.py``` Flask app logic + calling openpose and dgnn library
2. ```data_gen/``` Data preparation for DGNN (adapted from the repo as cited below)

Will clean up the code and do proper documentation soon (and probably should dockerize it)

## Citations

* Pose Estimation framework: OpenPose https://github.com/CMU-Perceptual-Computing-Lab/openpose
* Video activity recognition: Graph Convolution Network https://github.com/kenziyuliu/DGNN-PyTorch
<br>
More documentation will be provided, repo is full of hardcoded stuff.
