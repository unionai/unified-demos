import os
import flytekit
from flytekit import Resources, task, workflow
import flytekit.deck
from flytekitplugins.spark import Spark
from flytekitplugins.spark import DatabricksV2 as Databricks
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
import plotly.graph_objects as go
import numpy as np
import csv
from flytekit.types.file import CSVFile
from union import FlyteFile
import tempfile
from flytekit.image_spec import ImageSpec
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import wget

def download_file(url, save_path):
    try:
        wget.download(url, save_path)
    except Exception as e:
        print(f"Error downloading file: {e}")

SPARK_RUNTIME = "spark"

spark_conf = {
    "spark.driver.memory": "1000M",
    "spark.executor.memory": "1000M",
    "spark.executor.cores": "1",
    "spark.executor.instances": "2",
    "spark.driver.cores": "1",
    "spark.jars": "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar",
}

image_spec = ImageSpec(
    builder="union",
    name="spark",
    requirements="requirements-spark.txt"
)

task_config = Spark(spark_conf=spark_conf)

@task(cache=False, cache_version="1",
      container_image=image_spec)
def get_data() -> FlyteFile:

    data_url = "https://raw.githubusercontent.com/selva86/datasets/master/Iris.csv"
    ff = FlyteFile(path=data_url)
    return ff


@task(
    task_config=task_config,
    limits=Resources(mem="2000M"),
    container_image=image_spec,
)
def train_random_forest(data_file: FlyteFile) -> int:
    spark = flytekit.current_context().spark_session
    
    df = spark.read.csv(data_file.download(), header=True, inferSchema=True) 
    df.show(5)
    return 0

@workflow
def train_rf() -> FlyteFile:
    data_file = get_data()
    rf = train_random_forest(data_file)
    return data_file
