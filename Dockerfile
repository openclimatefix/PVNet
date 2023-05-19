FROM python:3.10-slim

ARG TESTING=0

# make sure it doesnt fail if the docker file doesnt know the git commit
ARG GIT_PYTHON_REFRESH=quiet

RUN apt-get update
RUN apt-get install git -y

# copy files
COPY setup.py app/setup.py
COPY README.md app/README.md
COPY requirements.txt app/requirements.txt
RUN pip install git+https://github.com/SheffieldSolar/PV_Live-API#pvlive_api


# install requirements
RUN pip install -r app/requirements.txt

# copy library files
COPY pvnet/ app/pvnet/
COPY tests/ app/tests/

# change to app folder
WORKDIR /app

# install library
RUN pip install -e .

RUN if [ "$TESTING" = 1 ]; then pip install pytest pytest-cov coverage; fi

CMD ["python", "-u","pvnet/app.py"]
