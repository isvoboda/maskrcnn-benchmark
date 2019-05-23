# Maskrcnn benchmark in Innovatrics

## Training

Follow the main [README.md](../README.md).
Data in COCO format are available at `/mnt/nas.brno/image_tasks/1552_fingerpint_fore_back_segmentation/1586_sausage_fingerprints`.
The default backbone was deployed based on `configs/e2e_mask_rcnn_R_50_FPN_1x.yaml`



## Inference

Start `lab` in the docker container environment and copy the generated url including the token into your browser.
Start the lab service

~~~bash
cd docker
docker-compose run --rm --service-ports lab
~~~

In lab go to `demo` folder and launch the `slap-inference.ipynb`.


## Development in VSCode and Jupyter Demo

Follows a brief description of how to develop in the docker environment and how to run the demo with slightly modified Dockerfile in `dev/Dockerfile`.

## VScode Remote-Containers

Currently, 2019. 10. 05., is the environment based on container supported in **VSCode insiders** only.

Docker based development in VSCode needs a configuration files `devcontainer.json` and `docker-compose.yml` to be put into a `.devcontainer` folder and reopen the project with Remote-Containers localy.

**`devcontainer.json`** configures what docker-compose files should be used.

~~~json
// See https://aka.ms/vscode-remote/containers for the
// documentation about the devcontainer.json format
{
	"name": "Existing Docker Compose (Extend)",
	"dockerComposeFile": [
		"../docker/docker-compose.yml",
		"docker-compose.yml",
	],
	"service": "dev",
	"workspaceFolder": "/app",
	"extensions": [
		"ms-python.python",
	],
	"shutdownAction": "stopCompose"
}
~~~

**`docker-compose.yml`** binds the current user .gitconfig

~~~yaml
#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

version: '3'
services:
  # Update this to the name of the service you want to work with in your docker-compose.yml file
  dev:
    # Uncomment if you want to add a different Dockerfile in the .devcontainer folder
    # build:
    #   context: .
    #   dockerfile: Dockerfile

    # Uncomment if you want to expose any additional ports. The snippet below exposes port 3000.
    # ports:
    #   - 3000:3000
    
    volumes:
    #  # Update this to wherever you want VS Code to mount the folder of your project
    #  - ..:/workspace

      # This lets you avoid setting up Git again in the container
      - ~/.gitconfig:/root/.gitconfig
      - ${SSH_AUTH_SOCK}:/ssh-agent

      # Forwarding the socket is optional, but lets docker work inside the container if you install the Docker CLI.
      # See the docker-in-docker-compose definition for details on how to install it.
      # - /var/run/docker.sock:/var/run/docker.sock 
    environment:
      - SSH_AUTH_SOCK=/ssh-agent

    # Overrides default command so things don't shut down after the process ends - useful for debugging
    command: sleep infinity 
~~~
