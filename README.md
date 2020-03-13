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
