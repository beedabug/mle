# base image
FROM python:3.11

# set the working directory for containers
# create and set the working directory within the image
WORKDIR /api

# copy files from cwd to the image (COPY <src> <dest>)
COPY api/data/exercise_26_train.csv data/exercise_26_train.csv
COPY api/data/exercise_26_test.csv data/exercise_26_test.csv
COPY api/code/train.py code/train.py
COPY api/code/data_prep.py code/data_prep.py
COPY api/model/glm.joblib model/glm.joblib
COPY api/app.py app.py
COPY api/requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# run python scripts inside the image
RUN python code/train.py
RUN python code/data_prep.py

# set the command to run the Flask app
CMD ["python", "app.py"]