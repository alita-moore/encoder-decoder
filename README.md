# encoder-decoder

An image to text encoder decoder built using pytorch

# Instructions

You will need python 3.10 and poetry installed to run this project. I recommend that you run it in a devcontainer, though. 

## Devcontainer setup
First, allocate an ubuntu (22.04) vm instace with gpu and ssh into it.

1. create the working directory: `sudo mkdir /workspace && cd /workspace`
2. clone the repository `sudo git clone https://github.com/alita-moore/img-to-text && cd img-to-text`
3. setup the vm `sudo bash .devcontainer/setup-vm.sh` (you'll need to press enter / yes during the process)
4. Restart your machine: `sudo shutdown -r now` (you'll need to ssh back into the system)
5. cd into the project directory `cd /workspace/img-to-text`
6. run `docker login` and login to docker, this is necessary to pull the relevant cuda image
7. build and launch the devcontainer `devcontainer up --workspace-folder . --remove-existing-container`
8. setup a local docker context and connect to the running devcontainer remotely via vscode (tutorial: https://www.doppler.com/blog/visual-studio-code-remote-dev-containers-on-aws)
9. once inside of the container navigate to /workspaces/img-to-text and run `poetry install`

## Running the model

You can test inference capabilities via the dev.py file which mimics a jupyter notebook. To collect torch trace logs you should run the model with the following command:

`TORCH_TRACE=/logs poetry run python dev.py`

## Local setup

If you wish to run this code locally instead, make sure you have at least cuda 12.4 installed and then run `poetry install`. You can install poetry via `pip install pipx && pipx install poetry && pipx ensurepath` if it's not already installed.

# Acknowledgments

The decoder code was largely provided by https://github.com/EleutherAI/aria-amt
