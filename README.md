# TimeOverflow

## Setup
```sh
docker build . -t timeoverflow_data
```
## Run
```sh
docker run --name to_data -e TO_DB_SERVER=<server> -e TO_DB_USER=<user> -e TO_DB_PASSWORD=<pass> --network="host" timeoverflow_data
```
## Check for results
```sh
docker inspect -f '{{ .Mounts }}' to_data
```

## Periodic launch of the script
Each time the container is run, it launches the script until the end and then
quits, i.e. the docker container is stopped. To relaunch the script simply:
```sh
docker start to_data
```

This re-runs the scripts quietly (no stdout), and rewrites the files to the
designated docker volume and at the end stops the docker container.
