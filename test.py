import subprocess
from demos.basic.actors.visual import BasicVisual,BasicCaimanVisual
from improv.link import Link
import sys

def run(name1, name2, control_port):
    # args needs name
    b = BasicCaimanVisual(name2)
    a = BasicVisual(name1)
    a.setup(visual=b)
    a.run(control_port=control_port)

if __name__ == '__main__':
    # name1 and name2 are cmd args
    run(sys.argv[1], sys.argv[2], sys.argv[3])

