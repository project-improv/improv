import logging

# Matplotlib is overly verbose by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == "__main__":
    from improv.nexus import Nexus

    loadFile = "demos/bubblewrap/bubble_demo.yaml"

    nexus = Nexus("Nexus")
    nexus.createNexus(file=loadFile)

    nexus.startNexus()
