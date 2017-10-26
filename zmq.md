## ZEROMQ

ZeroMQ is a light weight, super-fast socket communication library for which wrappers are available in many languages. Among others, in Python, JavaScript, C++, etc.
The main Web page http://zeromq.org/ provides a wealth of information. Among others, you find the official guide there for different languages: http://zguide.zeromq.org/py:all
The Python library to be used is pyzmq. Its home is under https://github.com/zeromq/pyzmq. Install it via conda install pyzmq or pip install pyzmq.
Examples for projects/institutions which use ZeroMQ as their messaging protocol are Jupyter Notebook, LHC, NFL and many more.
The following code uses two (non-standard) libraries, autobahn and Twisted. I can install them via:
   
    pip install autobahn
    pip install Twisted
