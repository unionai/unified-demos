import union
from union.artifacts import OnArtifact
from unified_wfs import UnifiedTrainedModel
from unified_deploy_wfs import ModelProductionTestResultsArtifact
from unified_deploy_wfs import test_model_for_production
from unified_deploy_wfs import promote_to_prod_wf

trigger_on_trained_model = OnArtifact(
    trigger_on=UnifiedTrainedModel.query(environment="dev"),
)

trigger_on_test_results = OnArtifact(
    trigger_on=ModelProductionTestResultsArtifact,
)

downstream_triggered = union.LaunchPlan.create(
    "unified_evaluate_model_for_prod",
    test_model_for_production,
    trigger=trigger_on_trained_model,
    auto_activate=True
)

downstream_triggered = union.LaunchPlan.create(
    "unified_promote_to_prod",
    promote_to_prod_wf,
    trigger=trigger_on_test_results,
    auto_activate=True
)
