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
import tempfile
from flytekit.image_spec import ImageSpec
import random
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


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


@task(
    task_config=task_config,
    limits=Resources(mem="2000M"),
    enable_deck=True,
    container_image=image_spec,
)
def train_rf(dataset_file: CSVFile) -> int:
    sess = flytekit.current_context().spark_session
    dataset = sess.read.csv(dataset_file.remote_source, header=True, inferSchema=True)
    dataset = dataset.withColumn("label", dataset["label"].cast('int'))
    dataset.show(5)

    assembler = VectorAssembler(inputCols=["x", "y", "z"], outputCol="features")

    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    # Split the data into training and test sets
    train_data, test_data = dataset.randomSplit([0.7, 0.3], seed=42)

    pipeline = Pipeline(stages=[assembler, rf])

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20, 30]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .build()

    # Create the cross-validator
    cross_validator = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"),
                            numFolds=5, seed=42)

    # Train the model with the best hyperparameters
    cv_model = cross_validator.fit(train_data)

    print(dataset)
    print(cv_model)

    predictions = cv_model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

    # Evaluate the model
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = {:.2f}".format(accuracy))

    return 0


@task(enable_deck=True, 
      cache=False, cache_version="1",
      container_image=image_spec)
def generate_data(dataset_size: int) -> CSVFile:
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate 50 points around (1, 1, 1) with a smaller spread
    cluster_1 = np.random.normal(loc=1, scale=0.2, size=(int(dataset_size/2), 3))

    # Generate 50 points around (3, 3, 3) with a smaller spread
    cluster_2 = np.random.normal(loc=3, scale=0.2, size=(int(dataset_size/2), 3))

    # Combine the clusters to form the dataset
    coordinates = np.vstack((cluster_1, cluster_2))

    # Create a temporary file to store the CSV data
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv')
    try:
        writer = csv.writer(temp_file)
        # Write the header
        writer.writerow(['x', 'y', 'z', 'label'])
        # Write the coordinates
        for row in coordinates:
            label = random.randint(1, 10)
            row = np.append(row, label)
            writer.writerow(row)
    finally:
        temp_file.close()

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
                               mode='markers',
                               marker=dict(size=5,
                                           color=np.concatenate([np.zeros(50), np.ones(50)]),  # Example cluster labels
                                           colorscale='Viridis',
                                           opacity=0.8),
                               name='Data Points'))
    
    # Adjust layout for full page
    fig.update_layout(autosize=True, margin=dict(l=0, r=0, b=0, t=0))

    flytekit.Deck("Visualizer", fig.to_html(full_html=False, include_plotlyjs='cdn'))

    return CSVFile(path=temp_file.name)


@workflow
def train(dataset_size: int = 100) -> int:
    dataset = generate_data(dataset_size=dataset_size)
    clusters = train_rf(dataset_file=dataset)
    return clusters
