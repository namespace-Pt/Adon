Welcome to Adon's documentation!
=========================================

.. note::

   This project is under active development. The docs are not complete.


Adon is an all-in-one python framework for **Ad**-h**o**c I**n**formation Retrieval. It is highlighted for the following properties:

1. **All-in-one.** Adon efficiently implements the entire ad-hoc retrieval pipeline: including the data loading, model training, encoding, indexing, and evaluation. Based on the this, Adon implements various neural retrieval/reranking models (Sparse, Dense, the newly proposed Generative, and the cross-encoder). One can easily train/evaluate a model with one line of command.
2. **Elastic.** The components of Adon are carefully designed to be highly extendable and independent with one another. You can focus on developing a specific part (e.g. encoding model) without inspecting details about the others (e.g. implementation of ANN indexes). You can also develop your own model within the Adon's framework by only adding a configuration file and a model's implementation.

Before you start, make sure you installed the dependencies:

.. code-block:: console

   export CUDA=11.6
   conda create -n adon python=3.9.12
   conda install pytorch==1.12.1 cudatoolkit=$CUDA -c conda-forge -c pytorch
   conda install faiss-gpu==1.7.2 -c conda-forge
   pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0+$CUDA.html
   pip install transformers==4.21.3 hydra-core==1.2.0 notebook ipywidgets psutil

.. note::

   Make sure the cudatoolkit version matches your machine!

Then clone the repository:

.. code-block:: console

   git clone https://github.com/namespace-Pt/Adon
   cd Adon
   git lfs pull

Now, let's start the journey.

Contents
--------
..
   include all the markdown file in the experiments folder with the experiments as top-title

.. toctree::
   :maxdepth: 2

   Home <self>
   sources/guides/index
   sources/utils/index
