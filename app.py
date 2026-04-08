#!/usr/bin/env python3
"""
Streamlit web interface for Point Cloud Classification System.
"""
import streamlit as st
import os
import sys
import tempfile
import numpy as np
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="Point Cloud Classification",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("☁️ Point Cloud Classification System")
st.markdown("""
This web interface allows you to interact with the Point Cloud Classification System.
You can visualize point clouds, run inference, and compare models.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Visualize Point Clouds", "Model Inference", "Model Comparison", "Training"]
)

# Home page
if page == "Home":
    st.header("Welcome!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Features")
        st.markdown("""
        - **Multiple Models**: Point Transformer, PointNet, PointNet++, DGCNN
        - **Dataset Support**: ScanObjectNN, S3DIS
        - **Visualization**: 3D point cloud visualization
        - **Training**: Full training pipeline
        - **Evaluation**: Comprehensive metrics
        """)

    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. Install dependencies: `pip install -r requirements.txt`
        2. Download dataset: `python main.py download-scanobjectnn`
        3. Train model: `python main.py train --model point_transformer`
        4. Run this app: `streamlit run app.py`
        """)

    st.subheader("📈 System Status")

    # Check dependencies
    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        try:
            import torch
            st.success(f"PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                st.info(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.warning("GPU: Not available")
        except ImportError:
            st.error("PyTorch: Not installed")

    with status_col2:
        try:
            import streamlit
            st.success(f"Streamlit: {streamlit.__version__}")
        except ImportError:
            st.error("Streamlit: Not installed")

    with status_col3:
        try:
            import plotly
            st.success(f"Plotly: {plotly.__version__}")
        except ImportError:
            st.error("Plotly: Not installed")

# Visualize Point Clouds page
elif page == "Visualize Point Clouds":
    st.header("Visualize Point Clouds")

    # Dataset selection
    dataset = st.selectbox(
        "Select Dataset",
        ["ScanObjectNN", "S3DIS", "Upload Custom"]
    )

    if dataset == "Upload Custom":
        uploaded_file = st.file_uploader(
            "Upload point cloud file (CSV, TXT, NPY, PCD)",
            type=["csv", "txt", "npy", "pcd"]
        )

        if uploaded_file is not None:
            # Read file based on extension
            file_ext = uploaded_file.name.split('.')[-1].lower()

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Parse point cloud
            try:
                if file_ext == "csv" or file_ext == "txt":
                    points = np.loadtxt(tmp_path, delimiter=',')
                elif file_ext == "npy":
                    points = np.load(tmp_path)
                elif file_ext == "pcd":
                    # Simple PCD parser for demonstration
                    import pandas as pd
                    with open(tmp_path, 'r') as f:
                        lines = f.readlines()

                    # Find DATA section
                    data_start = None
                    for i, line in enumerate(lines):
                        if "DATA" in line:
                            data_start = i + 1
                            break

                    if data_start is not None:
                        points = np.loadtxt(lines[data_start:])
                    else:
                        st.error("Could not parse PCD file")
                        points = None
                else:
                    st.error(f"Unsupported file format: {file_ext}")
                    points = None

                if points is not None:
                    # Ensure points have correct shape
                    if points.shape[1] > points.shape[0]:
                        points = points.T

                    # Take only x, y, z coordinates
                    if points.shape[0] > 3:
                        points = points[:3, :]

                    # Downsample if too many points
                    if points.shape[1] > 10000:
                        indices = np.random.choice(points.shape[1], 10000, replace=False)
                        points = points[:, indices]

                    # Visualize
                    fig = go.Figure(data=[
                        go.Scatter3d(
                            x=points[0, :],
                            y=points[1, :],
                            z=points[2, :],
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=points[2, :],  # Color by z-coordinate
                                colorscale='Viridis',
                                opacity=0.8
                            )
                        )
                    ])

                    fig.update_layout(
                        title=f"Point Cloud: {uploaded_file.name}",
                        scene=dict(
                            xaxis_title="X",
                            yaxis_title="Y",
                            zaxis_title="Z",
                            aspectmode="data"
                        ),
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Points", points.shape[1])
                    with col2:
                        st.metric("X Range", f"{points[0, :].min():.2f} - {points[0, :].max():.2f}")
                    with col3:
                        st.metric("Y Range", f"{points[1, :].min():.2f} - {points[1, :].max():.2f}")

            except Exception as e:
                st.error(f"Error loading file: {e}")

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    else:
        # For predefined datasets
        num_points = st.slider("Number of points to display", 100, 5000, 1000)
        sample_idx = st.number_input("Sample index", min_value=0, max_value=1000, value=0)

        if st.button("Visualize Sample", type="primary"):
            with st.spinner("Loading point cloud..."):
                # Create random point cloud for demonstration
                # In real application, load from dataset
                points = np.random.randn(3, num_points) * 0.5

                # Add some structure
                points[0, :] += np.sin(np.linspace(0, 4*np.pi, num_points))
                points[1, :] += np.cos(np.linspace(0, 4*np.pi, num_points))

                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=points[0, :],
                        y=points[1, :],
                        z=points[2, :],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=points[2, :],
                            colorscale='Plasma',
                            opacity=0.8
                        )
                    )
                ])

                fig.update_layout(
                    title=f"{dataset} Sample #{sample_idx}",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        aspectmode="data"
                    ),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

# Model Inference page
elif page == "Model Inference":
    st.header("Model Inference")

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Select Model",
            ["Point Transformer", "PointNet", "PointNet++", "DGCNN"]
        )

        checkpoint_file = st.file_uploader(
            "Upload model checkpoint (optional)",
            type=["pth", "pt"]
        )

    with col2:
        num_classes = st.slider("Number of classes", 2, 50, 15)
        num_points = st.slider("Number of points", 256, 4096, 1024)

    # Generate random point cloud for inference
    if st.button("Run Inference", type="primary"):
        with st.spinner("Running inference..."):
            # In real application, load model and run inference
            # Here we simulate with random predictions

            import torch
            import time

            # Simulate model loading
            time.sleep(1)

            # Generate random predictions
            predictions = torch.randn(1, num_classes)
            probabilities = torch.softmax(predictions, dim=1)

            # Get top predictions
            top_probs, top_classes = torch.topk(probabilities, k=3)

            # Display results
            st.subheader("Results")

            cols = st.columns(3)
            for i, (col, prob, class_idx) in enumerate(zip(cols, top_probs[0], top_classes[0])):
                with col:
                    class_name = f"Class {class_idx.item()}"
                    if hasattr(st, 'metric'):
                        st.metric(
                            label=class_name,
                            value=f"{prob.item():.1%}",
                            delta="confident" if prob.item() > 0.5 else "uncertain"
                        )
                    else:
                        st.write(f"**{class_name}**: {prob.item():.1%}")

            # Confidence chart
            import plotly.express as px

            class_indices = list(range(num_classes))
            prob_values = probabilities[0].detach().numpy()

            fig = px.bar(
                x=class_indices,
                y=prob_values,
                labels={'x': 'Class', 'y': 'Probability'},
                title="Class Probabilities"
            )

            st.plotly_chart(fig, use_container_width=True)

# Model Comparison page
elif page == "Model Comparison":
    st.header("Model Comparison")

    models_to_compare = st.multiselect(
        "Select models to compare",
        ["Point Transformer", "PointNet", "PointNet++", "DGCNN", "Random Forest", "SVM"],
        default=["Point Transformer", "PointNet", "DGCNN"]
    )

    metric = st.selectbox(
        "Comparison metric",
        ["Accuracy", "Precision", "Recall", "F1-Score", "Inference Time"]
    )

    if st.button("Compare Models", type="primary"):
        # Generate comparison data
        import pandas as pd
        import plotly.express as px

        # Simulate comparison results
        data = []
        for model in models_to_compare:
            if "Transformer" in model:
                accuracy = 0.925 + np.random.randn() * 0.02
                inference_time = 120 + np.random.randn() * 10
            elif "PointNet" in model:
                accuracy = 0.892 + np.random.randn() * 0.02
                inference_time = 80 + np.random.randn() * 5
            elif "DGCNN" in model:
                accuracy = 0.915 + np.random.randn() * 0.02
                inference_time = 150 + np.random.randn() * 15
            else:
                accuracy = 0.75 + np.random.rand() * 0.15
                inference_time = 50 + np.random.rand() * 30

            data.append({
                "Model": model,
                "Accuracy": max(0, min(1, accuracy)),
                "Precision": max(0, min(1, accuracy - 0.05 + np.random.rand() * 0.1)),
                "Recall": max(0, min(1, accuracy - 0.03 + np.random.rand() * 0.08)),
                "F1-Score": max(0, min(1, accuracy - 0.02 + np.random.rand() * 0.06)),
                "Inference Time": max(10, inference_time),
                "Parameters (M)": np.random.randint(3, 15)
            })

        df = pd.DataFrame(data)

        # Display table
        st.subheader("Comparison Results")
        st.dataframe(df, use_container_width=True)

        # Visualization
        fig = px.bar(
            df,
            x="Model",
            y=metric,
            color="Model",
            title=f"Model Comparison by {metric}",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot: Accuracy vs Inference Time
        fig2 = px.scatter(
            df,
            x="Inference Time",
            y="Accuracy",
            size="Parameters (M)",
            color="Model",
            hover_name="Model",
            title="Accuracy vs Inference Time",
            height=400
        )

        st.plotly_chart(fig2, use_container_width=True)

# Training page
elif page == "Training":
    st.header("Model Training")

    with st.form("training_config"):
        col1, col2 = st.columns(2)

        with col1:
            model = st.selectbox(
                "Model",
                ["Point Transformer", "PointNet", "PointNet++", "DGCNN"]
            )

            dataset = st.selectbox(
                "Dataset",
                ["ScanObjectNN", "S3DIS", "Custom"]
            )

            epochs = st.slider("Epochs", 10, 500, 100)

        with col2:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-1,
                value=1e-3,
                format="%.5f"
            )

            batch_size = st.select_slider(
                "Batch Size",
                options=[8, 16, 32, 64, 128],
                value=32
            )

            use_augmentation = st.checkbox("Use Data Augmentation", value=True)
            use_early_stopping = st.checkbox("Use Early Stopping", value=True)

        submitted = st.form_submit_button("Start Training", type="primary")

        if submitted:
            # Simulate training
            import time
            from streamlit import empty

            progress_bar = st.progress(0)
            status_text = st.empty()

            for epoch in range(epochs):
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)

                # Simulate training metrics
                train_loss = 1.0 * np.exp(-epoch / 20) + np.random.rand() * 0.1
                val_accuracy = 0.1 + 0.8 * (1 - np.exp(-epoch / 30)) + np.random.rand() * 0.05

                status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {val_accuracy:.4f}")

                # Simulate epoch time
                time.sleep(0.05)

            # Show final results
            st.success(f"Training completed! Final accuracy: {val_accuracy:.4f}")

            # Plot training history
            import plotly.graph_objects as go

            epochs_range = list(range(1, epochs + 1))
            loss_history = [1.0 * np.exp(-e / 20) + np.random.rand() * 0.1 for e in epochs_range]
            acc_history = [0.1 + 0.8 * (1 - np.exp(-e / 30)) + np.random.rand() * 0.05 for e in epochs_range]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs_range,
                y=loss_history,
                mode='lines',
                name='Training Loss',
                line=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=epochs_range,
                y=acc_history,
                mode='lines',
                name='Validation Accuracy',
                yaxis='y2',
                line=dict(color='green')
            ))

            fig.update_layout(
                title="Training History",
                xaxis_title="Epoch",
                yaxis=dict(title="Loss", color="red"),
                yaxis2=dict(
                    title="Accuracy",
                    color="green",
                    overlaying="y",
                    side="right"
                ),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
Point Cloud Classification System v1.0.0

[GitHub Repository](https://github.com/ybyyb1/pointcloud-classification)
""")

if __name__ == "__main__":
    # This is already a Streamlit app
    pass