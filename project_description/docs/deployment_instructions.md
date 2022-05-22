## Download model weights

Download model weights via `initialize.sh`.

## Create DigitalOcean Droplet

`Terraform` is being used for the DigitalOcean droplet. Code for this is in `eyewash-terraform` folder.

- Log in via the droplet console and add your ssh key to ~/.ssh/authorized_keys. 
- SSH to the machine via `ssh root@159.65.37.174`.
- Copy code to the machine via 

`scp -r  eyewash root@159.65.37.174:eyewash/`

Alternatively, you can clone the repository directly there.

- Install docker via `install_docker.sh` (you might need to run the commmands 1 by 1)
- On digitalocean, go to firewall > inbound rules and open TCP Ports 8002 and 8504.

## Build Docker Images

- Build dashboard image

`cd dashboard`
`docker build -t eyewash_dashboard:latest .`

- Build API image

`cd app`
`docker build -t api_eyewash:latest .`

## Run service

`docker-compose up` in the root directory.

Access the API via `http://159.65.37.174:8002/docs` and the dsahboard via `http://159.65.37.174:8504/docs`.
