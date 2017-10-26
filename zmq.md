## ZEROMQ

ZeroMQ is a light weight, super-fast socket communication library for which wrappers are available in many languages. Among others, in Python, JavaScript, C++, etc.
The main Web page http://zeromq.org/ provides a wealth of information. Among others, you find the official guide there for different languages: http://zguide.zeromq.org/py:all
The Python library to be used is pyzmq. Its home is under https://github.com/zeromq/pyzmq. Install it via conda install pyzmq or pip install pyzmq.
Examples for projects/institutions which use ZeroMQ as their messaging protocol are Jupyter Notebook, LHC, NFL and many more.
The following code uses two (non-standard) libraries, autobahn and Twisted. I can install them via:
   
    pip install autobahn
    pip install Twisted
#### PUB-SUB

http://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/devices/forwarder.html

Below looks same of our ML client : Fowarder device
      
      import zmq
      def main():

          try:
              context = zmq.Context(1)
              # Socket facing clients
              frontend = context.socket(zmq.SUB)
              frontend.bind("tcp://*:5559")

              frontend.setsockopt(zmq.SUBSCRIBE, "")

              # Socket facing services
              backend = context.socket(zmq.PUB)
              backend.bind("tcp://*:5560")

              zmq.device(zmq.FORWARDER, frontend, backend)
          except Exception, e:
              print e
              print "bringing down zmq device"
          finally:
              pass
              frontend.close()
              backend.close()
              context.term()

      if __name__ == "__main__":
          main()

Below looks same of our ML client : Fowarder server. IB connector or main page

      import zmq
      import random
      import sys
      import time

      port = "5559"
      context = zmq.Context()
      socket = context.socket(zmq.PUB)
      socket.connect("tcp://localhost:%s" % port)
      publisher_id = random.randrange(0,9999)
      while True:
          topic = random.randrange(1,10)
          messagedata = "server#%s" % publisher_id
          print "%s %s" % (topic, messagedata)
          socket.send("%d %s" % (topic, messagedata))
          time.sleep(1)

#### Ordering and ML
We will be using PUB with IB connection[1]. Client will SUB that for ML[2] and again PUB to different port wich will be to [1] again for ordering
#### Details of the pack
1. IB_supporting_def.py [ this is for functions like loading, savings, preprocessing...]. This file to import
2. tpqib.py[ Ibpy wrapper to connect IB TWS]
3. Algorithmic Trading Class          
