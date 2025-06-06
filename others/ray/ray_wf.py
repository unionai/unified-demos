import typing

import ray
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

from flytekit import ImageSpec, Resources, task, workflow

custom_image = ImageSpec(
    # registry="ghcr.io/flyteorg",
    packages=["flytekitplugins-ray"],
    # kuberay operator needs wget for readiness probe.
    apt_packages=["wget"],
)

@ray.remote
def f(x):
    return x * x

ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(ray_start_params={"log-color": "True"}),
    worker_node_config=[WorkerNodeConfig(group_name="ray-group", replicas=1)],
    runtime_env={"pip": ["numpy", "pandas"]},  # or runtime_env="./requirements.txt"
    enable_autoscaling=True,
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=3600,
)

@task(
    task_config=ray_config,
    requests=Resources(mem="2Gi", cpu="2"),
    container_image=custom_image,
)
def ray_task(n: int) -> typing.List[int]:
    futures = [f.remote(i) for i in range(n)]
    return ray.get(futures)

@workflow
def ray_wf():
    ray_task(10)