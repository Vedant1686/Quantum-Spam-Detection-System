
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==================================================
# PAGE SETUP
# ==================================================

st.set_page_config(
    page_title="Quantum Spam Detection",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


# ==================================================
# LOAD TRAINED ARTIFACTS
# ==================================================

@st.cache_resource
def load_artifacts():
    classical_path = ARTIFACT_DIR / "classical_model.joblib"
    quantum_path = ARTIFACT_DIR / "quantum_model.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.joblib"

    for path in (classical_path, quantum_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    return (
        joblib.load(classical_path),
        joblib.load(quantum_path),
        joblib.load(metrics_path),
    )


try:
    classical_artifacts, quantum_artifacts, metrics = load_artifacts()
except Exception as exc:
    st.error("Could not load the model files.")
    st.code(str(exc))
    st.info(
        "Run `python train_model.py` first, then refresh this page."
    )
    st.stop()


# ==================================================
# PREDICTION HELPERS
# ==================================================

def predict_classical(message):
    vectorizer = classical_artifacts["vectorizer"]
    model = classical_artifacts["model"]

    features = vectorizer.transform([message])
    prediction = str(model.predict(features)[0])
    score = float(model.decision_function(features)[0])

    return prediction, score


def predict_quantum(message):
    vectorizer = quantum_artifacts["vectorizer"]
    reducer = quantum_artifacts["reducer"]
    scaler = quantum_artifacts["angle_scaler"]
    model = quantum_artifacts["model"]

    features = vectorizer.transform([message])
    reduced = reducer.transform(features)
    angles = scaler.transform(reduced)

    prediction = str(model.predict(angles)[0])
    score = float(model.decision_function(angles)[0])

    return prediction, score


def display_prediction(model_name, result):
    prediction, score = result

    if prediction == "spam":
        st.error("🚨 SPAM MESSAGE")
    else:
        st.success("✅ NOT SPAM (HAM)")

    st.caption(f"Model: {model_name}")
    st.caption(f"Decision score: {score:.4f}")


# ==================================================
# CHART HELPERS
# ==================================================

def confusion_matrix_figure(matrix, title):
    matrix = pd.DataFrame(
        matrix,
        index=["Actual Ham", "Actual Spam"],
        columns=["Predicted Ham", "Predicted Spam"],
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted label",
            "y": "Actual label",
            "color": "Messages",
        },
        title=title,
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=60, b=30),
    )

    return fig


def metric_comparison_figure(classical, quantum):
    rows = []

    for model_name, scores in [
        ("Classical SVM", classical),
        ("Quantum SVM", quantum),
    ]:
        for metric_name in [
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]:
            rows.append({
                "Model": model_name,
                "Metric": metric_name.title(),
                "Score": scores[metric_name],
            })

    data = pd.DataFrame(rows)

    fig = px.bar(
        data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=data["Score"].map(lambda value: f"{value:.1%}"),
        range_y=[0, 1],
        title="Classical vs Quantum Model Metrics",
        color_discrete_sequence=["#3984F5", "#A66CFF"],
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=430,
        yaxis_tickformat=".0%",
        legend_title_text="Model",
        margin=dict(l=20, r=20, t=70, b=30),
    )

    return fig


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(" Quantum Spam AI")
st.sidebar.caption("Hybrid Quantum-Classical ML")

page = st.sidebar.radio(
    "Navigation",
    [
        "Spam Detector",
        "Performance Dashboard",
        "Dataset Insights",
        "About Project",
    ],
)

model_choice = st.sidebar.selectbox(
    "Prediction model",
    [
        "Both Models",
        "Classical SVM",
        "Quantum SVM",
    ],
)

st.sidebar.divider()
st.sidebar.caption("SMS Spam Collection Dataset")


# ==================================================
# HEADER
# ==================================================

st.title(" Quantum Spam Detection")
st.write(
    "Analyze SMS messages with a classical machine learning model "
    "and a simulated quantum-kernel classifier."
)

st.caption(
    "Python · Streamlit · Scikit-learn · Qiskit"
)


# ==================================================
# PAGE 1: SPAM DETECTOR
# ==================================================

if page == "Spam Detector":

    st.subheader("Message Analyzer")

    message = st.text_area(
        "Enter an SMS or email message",
        placeholder=(
            "Example: Congratulations! You have won a free prize. "
            "Call now to claim your reward!"
        ),
        height=150,
        key="message_input",
    )

    analyze = st.button(
        "🔍 Analyze Message",
        type="primary",
        use_container_width=True,
    )

    if analyze:
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            with st.spinner("Analyzing message..."):

                if model_choice == "Classical SVM":
                    try:
                        result = predict_classical(message)
                        st.subheader("Prediction")
                        display_prediction("Classical Linear SVM", result)
                    except Exception as exc:
                        st.error(f"Classical prediction failed: {exc}")

                elif model_choice == "Quantum SVM":
                    try:
                        result = predict_quantum(message)
                        st.subheader("Prediction")
                        display_prediction("Quantum-kernel SVM", result)
                    except Exception as exc:
                        st.error(f"Quantum prediction failed: {exc}")

                else:
                    try:
                        classical_result = predict_classical(message)
                    except Exception as exc:
                        classical_result = None
                        st.error(f"Classical prediction failed: {exc}")

                    try:
                        quantum_result = predict_quantum(message)
                    except Exception as exc:
                        quantum_result = None
                        st.error(f"Quantum prediction failed: {exc}")

                    left, right = st.columns(2)

                    with left:
                        st.markdown("### Classical AI")
                        if classical_result:
                            display_prediction(
                                "Classical Linear SVM",
                                classical_result,
                            )

                    with right:
                        st.markdown("### Quantum AI")
                        if quantum_result:
                            display_prediction(
                                "Quantum-kernel SVM",
                                quantum_result,
                            )

                    if (
                        classical_result
                        and quantum_result
                        and classical_result[0] != quantum_result[0]
                    ):
                        st.warning(
                            "The models disagree. Consider the result "
                            "uncertain and review the message."
                        )

    st.divider()

    st.subheader("Project at a glance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset messages",
        f"{metrics['dataset_size']:,}",
    )
    col2.metric("Quantum qubits", "4")
    col3.metric("Models", "2")

    st.info(
        "The displayed decision score is not a calibrated probability. "
        "Neither model guarantees a correct classification."
    )


# ==================================================
# PAGE 2: PERFORMANCE DASHBOARD
# ==================================================

elif page == "Performance Dashboard":

    st.subheader("Model Performance")

    classical = metrics["classical"]
    quantum = metrics["quantum"]

    st.markdown("### Classical SVM")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{classical['accuracy']:.2%}")
    c2.metric("Spam precision", f"{classical['precision']:.2%}")
    c3.metric("Spam recall", f"{classical['recall']:.2%}")
    c4.metric("Spam F1-score", f"{classical['f1']:.2%}")

    st.caption(
        f"Test set: {classical['test_size']:,} messages · "
        f"{classical['test_spam_count']} actual spam messages"
    )

    st.markdown("### Quantum-kernel SVM")

    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Accuracy", f"{quantum['accuracy']:.2%}")
    q2.metric("Spam precision", f"{quantum['precision']:.2%}")
    q3.metric("Spam recall", f"{quantum['recall']:.2%}")
    q4.metric("Spam F1-score", f"{quantum['f1']:.2%}")

    st.caption(
        f"Test set: {quantum['test_size']:,} messages · "
        f"{quantum['test_spam_count']} actual spam messages"
    )

    st.divider()

    st.plotly_chart(
        metric_comparison_figure(classical, quantum),
        use_container_width=True,
    )

    st.subheader("Confusion Matrices")

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            confusion_matrix_figure(
                classical["confusion_matrix"],
                "Classical SVM",
            ),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            confusion_matrix_figure(
                quantum["confusion_matrix"],
                "Quantum-kernel SVM",
            ),
            use_container_width=True,
        )

    st.caption(
        "Matrix rows represent actual labels; columns represent "
        "predicted labels. The order is ham, then spam."
    )

    st.warning(
        "These results use different test sets: the classical model "
        "uses the full held-out test set, while the quantum model uses "
        "a 100-message subset. Compare them cautiously; this is not a "
        "like-for-like evaluation."
    )


# ==================================================
# PAGE 3: DATASET INSIGHTS
# ==================================================

elif page == "Dataset Insights":

    st.subheader("SMS Dataset Overview")

    distribution = metrics["dataset_distribution"]

    total = sum(distribution.values())

    c1, c2, c3 = st.columns(3)

    c1.metric("Total messages", f"{total:,}")
    c2.metric("Ham messages", f"{distribution['ham']:,}")
    c3.metric("Spam messages", f"{distribution['spam']:,}")

    distribution_df = pd.DataFrame({
        "Label": ["Ham", "Spam"],
        "Messages": [
            distribution["ham"],
            distribution["spam"],
        ],
    })

    left, right = st.columns(2)

    with left:
        pie = px.pie(
            distribution_df,
            names="Label",
            values="Messages",
            hole=0.45,
            title="Dataset Class Distribution",
            color="Label",
            color_discrete_map={
                "Ham": "#3984F5",
                "Spam": "#FF6B6B",
            },
        )
        st.plotly_chart(pie, use_container_width=True)

    with right:
        bar = px.bar(
            distribution_df,
            x="Label",
            y="Messages",
            text="Messages",
            title="Message Counts",
            color="Label",
            color_discrete_map={
                "Ham": "#3984F5",
                "Spam": "#FF6B6B",
            },
        )
        st.plotly_chart(bar, use_container_width=True)

    ham_percent = distribution["ham"] / total * 100
    spam_percent = distribution["spam"] / total * 100

    st.markdown("### Class balance")

    st.write(f"Ham messages: **{ham_percent:.2f}%**")
    st.write(f"Spam messages: **{spam_percent:.2f}%**")

    st.warning(
        "The dataset is imbalanced: legitimate messages are much more "
        "common than spam. Accuracy alone can therefore be misleading. "
        "Precision, recall and F1-score are important."
    )


# ==================================================
# PAGE 4: ABOUT PROJECT
# ==================================================

elif page == "About Project":

    st.subheader("About the Project")

    st.write(
        "This project compares a classical text classifier with a "
        "hybrid quantum-classical classifier for SMS spam detection."
    )

    st.markdown("### System workflow")

    st.markdown(
        """
        1. Load the labelled SMS Spam Collection dataset.
        2. Transform message text into TF-IDF features.
        3. Train a classical Linear SVM baseline.
        4. Reduce text features to four dimensions for quantum encoding.
        5. Calculate a quantum kernel using a simulated four-qubit circuit.
        6. Classify messages with a quantum-kernel SVM.
        7. Compare results with precision, recall, F1-score and confusion matrices.
        """
    )

    st.markdown("### Technologies")

    st.markdown(
        """
        - Python
        - Streamlit
        - Pandas and NumPy
        - Scikit-learn
        - Qiskit and Qiskit Machine Learning
        - Plotly
        """
    )

    st.markdown("### Limitations")

    st.write(
        "The quantum circuit runs on a classical simulator, not on "
        "physical quantum hardware. The quantum model uses a smaller "
        "training subset for practical simulation time. Its current "
        "performance is preliminary and does not establish an advantage "
        "over the classical model."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "B.Tech CSE Project | Quantum Computing + Artificial Intelligence"
)
