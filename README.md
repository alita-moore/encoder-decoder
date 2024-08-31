# encoder-decoder

An image to text encoder decoder built using pytorch

# Instructions

You will need python 3.10 and poetry installed to run this project. I recommend that you run it in a devcontainer, though. 

To do so first allocate an ubuntu (22.04) vm instace with gpu, ssh into it and then clone this repository, setup the vm using the setup script at `bash .devcontainer/setup-vm.sh`. 

To use run `poetry install` from the root directory. If you don't have poetry installed you can do so by running the following command: `pip install pipx && pipx install poetry && pipx ensurepath`.

# Acknowledgments

The decoder code was largely provided by https://github.com/EleutherAI/aria-amt