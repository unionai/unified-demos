import union
from union import Resources
from unified_wfs import UnifiedTrainedModel


image = union.ImageSpec(
    name="streamlit-app",
    packages=[
        "union-runtime>=0.1.10", "streamlit==1.41.1",
        "scikit-learn", "datasets", "pandas", "union>=0.1.145",
        "flytekit>=1.15.0",
        "seaborn", "matplotlib", "ing-theme-matplotlib"
        ],
    registry="pablounionai",
)

app_threshold = union.app.App(
    name="UnifiedDemos-threshold-app",
    inputs=[
        union.app.Input(
            name="UnifiedTrainedModel_threshold",
            value=UnifiedTrainedModel.query(),
            download=True,
            env_var="CLS_MODEL_RESULTS",
        ),
    ],
    container_image=image,
    args="streamlit run app_threshold.py --server.port 8080",
    port=8080,
    include=["."],
    limits=Resources(cpu="1", mem="1Gi"),
)