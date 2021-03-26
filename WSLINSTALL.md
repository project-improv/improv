# improv in Windows Subsystem for Linux

### This is a guide to installing and running improv on Windows systems using Windows Subsystem for Linux (WSL)

<br>

# Setup
## I. Install WSL
After completing the following WSL installation, the WSL shell can be activated by the `wsl` command in Command Prompt.
1. Enable WSL via Powershell as administrator
    ```
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
    ```
2. Restart system
3. Install Ubuntu 18.04 (or other distro) from [Windows Store](https://www.microsoft.com/en-us/p/ubuntu-1804-lts/9n9tngvndl3q?activetab=pivot%3Aoverviewtab)
4. Initialize distro 
    - Set username, password when prompted

WSL files can be accessed through the path: `~`, which is equivalent to `/home/[USER]`, within the WSL shell. Windows files, specifically the C: drive can be accessed through the path: `/mnt/c`. For RASP purposes, only the plasma store and the anaconda environment used to execute RASP is within the WSL system. The RASP code is located within Windows file system.

Furthermore, certain directories must be added to the PATH environmental variable in Linux in order for RASP to run properly. Below are the steps to "permanently" adding to the Linux $PATH.
1. Use vim to edit `~/.profile`
    ```
    vim ~/.profile
    ```
2. Add `export PATH="[PATH]:$PATH"` to end of `~/.profile`
3. Restart WSL shell

## II. Install Anaconda 3 in WSL
1. Find latest version of [Anaconda 3 for Linux](https://repo.continuum.io/archive)
2. Install latest version within WSL
    ```
    wget https://repo.continuum.io/archive/Anaconda3-[VERSION]-Linux-x86_64.sh
    ```
3. Run installation script
    ```
    bash Anaconda-[VERSION]-Linux-x86_64.sh
    ```
4. Opt to install Visual Studio Code when prompted
5. Update Anaconda if prompted or if necessary
    ```
    conda update anaconda
    ```
6. Add Anaconda to `$PATH` (see [Section I](#I.-Install-WSL))
    ```
    export PATH="~/anaconda3/bin:$PATH"
    ```

## III. Installing & Running X Server for RASP GUI
1. Download and install [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
2. Run XLaunch
    - Set Display number to "0"
    - Select Disable access control
    - Keep other default settings
3. Allow access through Windows firewall if prompted
4. Use vim to set display in `~/.profile` (See [Section 1](#I.-Install-WSL]))
    ```
    export DISPLAY=:0
    ```

## IV. RASP Installation
1. Clone improv
    ```
    git clone https://github.com/pearsonlab/improv
    ```
2. Clone RASP submodules
    ```
    git submodule update --init
    ```
3. Create Anaconda environment <br>
This creates a new anaconda environment named `caiman` using Python 3.6, which should be used for all future RASP installations and executions. Execute the following command in the CaImAn dir.
    ```
    conda env create -f environment.yml -n caiman
    conda activate caiman 
    ```
Alternatively, create an env with pre-installed dependencies within the improv/env directory.
    ```
    conda env create -f  improv_env.yml -n caiman
    conda activate caiman 
    ```

4. Install CaImAn module <br>
Execute the following command within the CaImAn directory and `caiman` env.
    ```
    pip install -e .
    ```
5. Install the following dependencies.
    - pyarrow
    - pyqtgraph
    ```
    conda install [PACKAGE]
    ```
6. Add pyarrow `plasma_store_server` to path (See [Section I](#I.-Install-WSL)) <br>
`plasma_store_server` is most likely located in the bin directory of the `caiman` conda environment: `~/anaconda3/envs/caiman/bin`. You can find where your environment is installed by entering `conda info --envs`. <br>


## V. Run improv
See [Common Issues](#Common-Issues) for errors and missing dependencies that might need to be installed.

1. Activate WSL
    ```
    wsl
    ```
2. Activate `caiman` environment 
    ```
    conda activate caiman
    ```
3. cd into improv/demos/basic
6. Run XLaunch (see [Section V](#V.-Installing-&-Running-X-Server-for-GUI-Framework)). The application may or may not appear on the task bar.
7. Run improv
    ```
    python basic_demo.py
    ```
   This demo will create its own data folder with the necessary data. The process should be replicated for future runs.

<br>

# Common Issues
## I. WSL Installation from Windows Store
If Ubuntu cannot be downloaded and installed from the Windows Store, it can be instead manually downloaded and installed through the following instructions.

1. Go to https://docs.microsoft.com/en-us/windows/wsl/install-manual to download distro or https://aka.ms/wsl-ubuntu-1804 to directly download Ubuntu 18.04
2. If running the .appx file downloaded does not successfully install, rename .appx extension to .zip and uncompress
3. Run `ubuntu18.04.exe` or altenerative distro executable
4. Complete initialization and rest of installation

## II. CaImAn/Anaconda Installation
Several issues can appear during the CaImAn installation process. It is recommended that if any severe issues appear, one should delete and then recreate the caiman conda environment from scratch. Or, caiman can be uninstalled (`pip uinstall caiman`) and installation can be reattempted. Some of the following issues might appear:

1. `Aborted (core dumped)` when activating the `caiman` env
    - Solved by updating all packages
        ```
        conda update --all
        ```
2. `Failed building wheel for caiman`: `command 'gcc' failed with exit status 1`
    - This error could be due to gcc or g++ not being installed. Follow these steps to install both. 
        ```
        conda update -all
        sudo apt-get update
        sudo apt-get install gcc
        sudo apt-get install g++
        ```

## III. Errors Running RASP

1. `ImportError: libGl.so.1: cannot open shared object file`
    - Solved by running the following
        ```
        sudo apt-get update
        sudo apt-get install libgl1-mesa-glx
        ```

2. `ImportError: libopencv_reg.so.3.4: cannot enable executable stack as shared object`
    - Solved by running the following (second line contains path of opencv installation)
        ```
        sudo apt-get update
        sudo apt-get install execstack
        sudo execstack -c ~/anaconda3/envs/caiman/lib/libopencv_*
        ```
3. `This application failed to start because it could not find or load the Qt platform plugin "xcb"`
    - Solved by running the following
        ```
        sudo apt-get update
        sudo apt-get install libqt5x11extras5
        ```
4. `gcc: error trying to exec 'cc1plus': execvp: No such file or directory`
    - Solved by installing gcc and g++
        ```
        conda update -all
        sudo apt-get update
        sudo apt-get install gcc
        sudo apt-get install g++
        ```

5. RASP freezing entirely
    - Solved by commented out `self.limbo.subscribe()` in nexus.py createNexus() function and `self.client.subscribe()` in store.py under the Watcher class

6. ModuleNotFound errors for specific packages
   - Solved by ensuring the following versions of packages were installed
   - scikit-learn==0.23.2
     numpy==1.18
     python <= 3.6



