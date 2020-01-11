FRAUDAR: Bounding Graph Fraud in the Face of Camouflage
Authors: Bryan Hooi, Hyun Ah Song, Alex Beutel, Neil Shah, Kijung Shin, Christos Faloutsos

This software is licensed under Apache License, Version 2.0 (the  "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Version: 1.1
Date: June 12, 2018
Main Contact: Bryan Hooi (bhooi@andrew.cmu.edu)


FRAUDAR is designed to catch fraudulent blocks in graph datasets (e.g. a review graph, Twitter follow graph, etc.) in a camouflage-resistant way. 

It detects blocks that are unusually dense while having low column degree.

============ RUNNING FRAUDAR ON A TEST EXAMPLE ============

The simplest way to run FRAUDAR is to use run_greedy.py, e.g.:

			python run_greedy.py data/example.txt out/out

This requires the python dependencies numpy and scipy (which can either be installed on their own or through a bundle like Anaconda)

The above command runs the greedy algorithm without node suspiciousness values on an example dataset (a sparse dataset with an injected 20 x 20 clique in the first 20 indices). As a result, the greedy algorithm should detect this 20 x 20 block. 

The first argument (data/example.csv) is the data file to use, while the second (out/out) is the location and filename to save the results. The output will be saved as "out/out.rows" and "out/out.cols", which are the set of rows and columns respectively in the block. 

*Optional*: If you want to include node suspiciousness values (i.e. obtained from some external source of information about how suspicious each node is), you can also add their path as a third argument: i.e.

	python run_greedy.py data/example.txt out/out data/susp

In this case the suspiciousness values (one value for each row and column) are stored as "data/susp.rows" and "data/susp.cols". 

A public amazon dataset, arts.csv used in the paper can be obtained from https://snap.stanford.edu/data/. The public twitter dataset used in the paper can be obtained from http://an.kaist.ac.kr/traces/WWW2010.html. 

============ FUNCTIONS IN GREEDY.PY ============

The main functions available for use in greedy.py are:

* readData: takes in the path to a file, which we read and then convert to a python sparse matrix. The input file should be a whitespace separated file of the form:

			(row_index) (col_index) (optional: 1)

(the "1" refers an edge weight of 1, but will be ignored since we assume an unweighted graph)

* aveDegree, sqrtWeightedAveDegree, and logWeightedAveDegree. For all of them, their first argument is the python sparse matrix of the form returned by readData. Their second *optional* argument is either None or a tuple (row_susp, col_susp), the node suspiciousness score vectors, which have length equal to the number of rows and columns respectively. We recommend the default of None (or not passing any 2nd argument), meaning no node suspiciousness, unless you have a clear idea of some external suspiciousness values to use. See run_greedy.py for an example. One possible use case is if the default is returning too large blocks and you want smaller and denser blocks, you can pass in constant positive vectors as node suspiciousness, which will tend to give smaller blocks. 

All these functions return a tuple ((row_indices, col_indices), score), i.e. the row and column indices of the detected block as well as the score value it receives. 

These functions only differ in the weighting scheme: aveDegree is unweighted (all edges have weight 1); sqrtWeightedAveDegree weights each edge by 1/sqrt(column degree) (i.e. column sum), while logWeightedAveDegree weights each weight by 1/log(column degree).

There are also some additional helper functions:

* subsetAboveDegree: remove all edges below a certain degree

* detectMultiple: for detecting multiple blocks

* shuffleMatrix: shuffle matrix rows and cols randomly

* injectCliqueCamo: inject a clique and various types of camouflage

* jaccard/getPrecision/getRecall/etc: various measures of the accuracy of the detected (row_indices, col_indices) tuple with respect to some ground truth tuple. 