# Outlier Description via Constraint Programming

Code in the repository implements a model proposed in my paper [A Framework for Outlier Description Using Constraint Programming](https://sites.google.com/site/chiatungkuo/publication).
The model aims to identify a feature subspace where a given set of known outliers have few neighbors whereas all normal instances have many neighbors.
We demonstrate this by a small example on the benchmark text data set, [20-Newsgroups](http://qwone.com/~jason/20Newsgroups/) (we used the preprocessed version for Matlab/Octave). This example corresponds to the second experiment in the paperand please referenec the paper for more information.

Files in this repository:

+ utils.py: includes a set of utility functions for preprocessing, such as get the most frequent words, construct document term matrices, etc.
+ preprocess.py: prepares the data to be used in the experiment and save them to disk.
+ demo.py: runs the example described in the paper using data generated from "preprocess.py"

Please set the path to files accordingly. The code assumes the user put all code in the directory "20news-bydate-matlab" decompressed from the archive file.

Dependency:

+ [Numberjack](http://numberjack.ucc.ie): We build our model using this programming platform. It has a simple interface in Python and also interfaces with state-of-the-art ILP solvers, such as Gurobi and CPLEX.
+ [Gurobi](http://www.gurobi.com): A state-of-the-art optimization program. We utilize its integer solver in particular with a free academic license.
+ [https://pypi.python.org/pypi/stop-words](https://pypi.python.org/pypi/stop-words): This is the set of stop-words we used in our experiment.

Please report any bugs/issues to [tomkuo@ucdavis.edu](mailto:tomkuo@ucdavis.edu).

