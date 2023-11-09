# ML_Zoomcamp_predict_brain_stroke

## About the project:
Are you worried about your or family member's health or chances of stroke and take precautionary steps?
Its is complicated to gather all the key information and check various sources to determine whether a patient is at risk of a brain stroke. With this model, you can provide a number of features and get an estimate of the chances of a brain stroke.


**This project was made for learning and fun purposes and is not a production service**

## About the data

The data is from the Brain Stroke Dataset in Kaggle.


## About the model
- Trained on 3984 patient records
- Used Random Forest
- Required attributes: `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`,
       `work_type`, `residence_type`, `avg_glucose_level`, `bmi`,
       `smoking_status` & `stroke`

## How to run the model
 - Download the files
 - Run train.py
 - Build Docker container: `docker build -t brain_stroke .`
 - Run Docker container: `docker run -it -p 9696:9696 brain_stroke:latest`


## Deploying to Beanstalk
- Setup a separate python virtual environment using `pipenv`

```
pipenv shell

pipenv install awsebcli --dev
```

- Test locally

```
eb init -p docker -r eu-west-1 stroke

eb local run --port 9696
```

- Create beanstalk environment

```
eb create stroke-env

eb terminate stroke-env
```