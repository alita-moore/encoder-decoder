# encoder-decoder

An image to text encoder decoder built using pytorch

# Instructions

You will need python 3.10 and poetry installed to run this project. I recommend that you run it in a devcontainer, though. 

1. create the working directory: `sudo mkdir \workspace && cd \workspace`
2. clone the repository `git clone https://github.com/alita-moore/img-to-text && cd img-to-text`
3. setup the vm `sudo bash .devcontainer/setup-vm.sh && source ~/.bashrc` (you'll need to press enter / yes during the process)
4. Reconnect your ssh connection to allow user permission updates to propogate
5. run `docker login` and login to docker, this is necessary to pull the relevant cuda image
6. build and launch the devcontainer `devcontainer up --workspace-folder .`

To do so first allocate an ubuntu (22.04) vm instace with gpu, ssh into it and then clone this repository, setup the vm using the setup script at `bash .devcontainer/setup-vm.sh`. 

To use run `poetry install` from the root directory. If you don't have poetry installed you can do so by running the following command: `pip install pipx && pipx install poetry && pipx ensurepath`.

# Acknowledgments

The decoder code was largely provided by https://github.com/EleutherAI/aria-amt
