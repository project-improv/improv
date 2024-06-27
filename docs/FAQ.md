(page:FAQ)=
# Frequently asked questions

#### I don't get it: what does _improv_ actually _do?_

Think of it like this: you're going to write code to run an adaptive experiment. That will likely do some combination of:
- gather data from your hardware
- store that data
- do any necessary preprocessing
- send the data through any required processing pipeline
- possibly run some models in the loop
- trigger stimulation, task conditions, or feedback based on the incoming data
- continuously plot something
- repeat

All of this is possible for a decent programmer and a single experiment. The complexity arises when you want to:
- change the data flow
- swap in or out steps in the pipeline
- handle data flow between different machines
- recover from crashes
- record anything you changed during the experiment
- use custom modeling

At that point, you are more or less inventing a small networking library for communication and data management. That's what _improv_ does.
You're welcome.

#### Will you be developing a graphical user interface (GUI) for _improv_?

At present there are no plans to build a general-purpose GUI with _improv_. There are two main reasons:
1. **GUIs are a maintenance nightmare.** There is no possible way for us to design an interface that will work for every experiment. 
Different combinations of recording and stimulation methods, model systems, and online analyses result in a huge number of possible
combinations &mdash; far more than we can support or anticipate. Rather, you should think of _improv_ as a set of building blocks
that can be flexibly combined to create a large range of experiments, rather than a single system designed for one type of 
pipeline.
1. **The GUI you build will best meet your needs.** While coding a GUI from scratch would be very hard, there are many GUI design
systems that allow users to build their own with only a small amount of code. For the demos in 
[our paper](https://www.biorxiv.org/content/10.1101/2021.02.22.432006) we used 
[PyQt](https://riverbankcomputing.com/software/pyqt), but there are [many other options](https://wiki.python.org/moin/GuiProgramming).
All that's required is some means of sending messages to and receiving them from the _improv_ server. Components onscreen can 
then be updated accordingly. 
