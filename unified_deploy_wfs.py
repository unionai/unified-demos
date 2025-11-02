import union
from typing import Annotated
from unified_wfs import UnifiedTrainedModel
from unified_wfs import image
from common_v1.common_dataclasses import HpoResults
from common_v1.common_dataclasses import ModelProductionTestResults
from flytekit import FlyteDirectory
from union.remote import UnionRemote
from union import Secret


# Configuration Parameters
environment = "dev"
production_environment = "prod"
decommission_environment = "decommissioned"

# Artifact Definitions
model_query = UnifiedTrainedModel.query(
    environment=environment)
model_query_prod = UnifiedTrainedModel.query(
    environment=production_environment)

ModelProductionTestResultsArtifact = union.artifacts.Artifact(
    name="unified_demo_model_production_test_results",
)
promotion_query = ModelProductionTestResultsArtifact.query()


# TASK definitions
@union.task(
    container_image=image,
)
def tsk_promote_model(target_model: HpoResults, to_environment: str)\
        -> Annotated[FlyteDirectory, UnifiedTrainedModel]:
    return UnifiedTrainedModel.create_from(
        target_model.to_flytedir(),
        target_model.get_model_card(),
        environment=to_environment
    )


@union.task(
    container_image=image,
)
def tsk_load_model(model_dir: FlyteDirectory) -> HpoResults:
    return HpoResults.from_flytedir(model_dir)


@union.task(
    container_image=image,
)
def tsk_test_model(target_model: HpoResults, prod_model: HpoResults)\
        -> Annotated[ModelProductionTestResults,
                     ModelProductionTestResultsArtifact]:
    curr_acc = target_model.acc - .04
    target_acc = target_model.acc
    promote_model = target_acc > curr_acc
    mpr = ModelProductionTestResults(
        promote_model=promote_model,
        target_model_acc=target_acc,
        current_prod_model_acc=curr_acc)
    return ModelProductionTestResultsArtifact.create_from(
        mpr, mpr.get_card()
    )


@union.task(
    container_image=image,
    secret_requests=[
        Secret(
            key="pablo-api-key", env_var="UNION_API_KEY",
            mount_requirement=Secret.MountType.ENV_VAR)]
)
def tsk_deploy_app_threshold():
    remote = UnionRemote()
    remote.deploy_app(app_threshold)


# WORKFLOW definitions
@union.workflow
def test_model_for_production(
        target_model_dir: FlyteDirectory = model_query,
        prod_model_dir: FlyteDirectory = model_query_prod
        ):
    target_model = tsk_load_model(target_model_dir)\
        .with_overrides(name="load_target_model")
    prod_model = tsk_load_model(prod_model_dir)\
        .with_overrides(name="load_prod_model")
    return tsk_test_model(target_model, prod_model)


@union.workflow
def promote_last_dev_to_prod_wf(
        target_model: FlyteDirectory = model_query
        ):
    model = tsk_load_model(target_model)\
        .with_overrides(name="load_target_model")
    return tsk_promote_model(model, production_environment)\
        .with_overrides(name="promote_to_prod")


@union.workflow
def promote_to_prod_wf(
        test_results: ModelProductionTestResults = promotion_query,
        curr_prod_model: FlyteDirectory = model_query_prod,
        target_model: FlyteDirectory = model_query
        ):
    model = tsk_load_model(target_model)\
        .with_overrides(name="load_target_model")
    curr_prod = tsk_load_model(curr_prod_model)\
        .with_overrides(name="load_curr_prod_model")
    prod = tsk_promote_model(model, production_environment)\
        .with_overrides(name="promote_to_prod")
    decom_model = tsk_promote_model(
        curr_prod, decommission_environment)\
        .with_overrides(name="decommission_curr_prod")
    return prod, decom_model


@union.workflow
def wf_deploy_app_threshold(art_query: HpoResults = model_query):
    tsk_deploy_app_threshold()
