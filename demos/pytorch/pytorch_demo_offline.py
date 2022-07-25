from actors.acquire_folder import FolderAcquirer

folder = "data/CIFAR10/images"

acq = FolderAcquirer("CIFAR10", folder=folder)
acq.run()


