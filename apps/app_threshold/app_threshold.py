import streamlit as st
from common.common_dataclasses import HpoResults
import os
from apps.app_threshold.model import get_predictions
from apps.app_threshold.eval import get_metrics_df
from apps.app_threshold.visualization import plot_probability_distribution
from apps.app_threshold.visualization import plot_confusion_matrix


obj = os.getenv("CLS_MODEL_RESULTS")
obj = HpoResults.from_flytedir(obj)


# data
@st.cache_data()
def cached_get_predictions():
    return get_predictions(obj)


y_train, yhat_prob_train, y_test, yhat_prob_test = cached_get_predictions()

# UI
st.title("Classifier Threshold Tuning")
threshold = st.slider(
    "Threshold", min_value=0.00, max_value=1.0, step=0.01, value=0.5)

# Metrics
metrics = get_metrics_df(
    y_train, yhat_prob_train, y_test, yhat_prob_test, threshold=threshold
)
# st.dataframe(metrics.assign(hack="").set_index("hack"))
st.dataframe(metrics)

if st.button("Deploy Model"):
    st.write("Model deployed to: "
             "https://mute-thunder-87bd9.apps.demo.hosted.unionai.cloud")

# Plots
cm = plot_confusion_matrix(yhat_prob_test, y_test, threshold)
st.pyplot(cm)

fig, ax = plot_probability_distribution(
    yhat_prob_train, y_train, threshold, "Train predictions"
)
st.pyplot(fig)

fig, ax = plot_probability_distribution(
    yhat_prob_test, y_test, threshold, "Test predictions"
)
st.pyplot(fig)