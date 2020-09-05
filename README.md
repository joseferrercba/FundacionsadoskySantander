# FundacionsadoskySantander

This information is related to Santander challenge:

[Clasificación de preguntas de clientes](https://metadata.fundacionsadosky.org.ar/competition/21)

## Create Environment From environment.yml
----------
Let's suppose that our new environment will have the following name: **env-03**
To create an environment from file, run this command:

```
conda env create -f environment.yml
```

If you want to create a new env. Execute this command:


```
conda create -n env-03 python=3.7.8
```

## Activate env
----------

To activate the environment you must run this command:
```
conda activate env-03
```

## Export env 
----------
`conda env export > environment.yml`

## Remove an env 
----------
`conda remove --name env-03 --all`

## Run mlflow
----------
`conda activate env-03`
`mlflow ui`