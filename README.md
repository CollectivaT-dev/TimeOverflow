# TimeOverflow

## Setup
To run the scripts it is necessary to setup the environment variables in the
.env file. First copy the `.env_default` to `.env`
```sh
cp .env_default .env
```

## Run
```sh
docker-compose build
docker-compose up
```

## Check for results
```sh
docker inspect -f '{{ .Mounts }}' to_data_1_<hash>
```

## Periodic launch of the script
Each time the container is run, it launches the script until the end and then
quits, i.e. the docker container is stopped. To relaunch the script simply:
```sh
docker-compose up
```

This re-runs the scripts quietly (no stdout), and rewrites the files to the
designated docker volume, pushes the data to the designated wo db and at the
end stops the docker container.
