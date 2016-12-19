End-To-End Memory Networks in Tensorflow applied to Sequential Modelling
========================================================================

This is a Tensorflow implementation of the [End-to-End Memory Network](http://arxiv.org/abs/1503.08895v4) applied to Sequential Modelling of Facebook comments.


Data
----

The data is already available in data.tar.gz and must be uncompressed in the data/ folder.


Running the code
----------------

To train the Memory Network, simply run the following command:

    $ python3 main.py
    
Different options can be set as arguments and are described as comments of the main.py file.


Evaluation
----------

We achieve the following performance:

| Network           | Perplexity   | 
| ----------------- | ------------ | 
| MN 				      	| 121.45       | 



Author
------

Rémi Vincent / [@remeus](https://github.com/Remeus)

