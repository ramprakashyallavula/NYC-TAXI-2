from datetime import timedelta
from typing import Optional

import pandas as pd
import plotly.express as px


from datetime import timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions=None,
):
    """
    Plots the time series data for a specific row in the features DataFrame.

    Args:
        features (pd.DataFrame): DataFrame containing feature data, including historical ride counts and metadata.
        targets (pd.Series): Series containing the target values (e.g., actual ride counts).
        row_id (int): Index of the row to plot.
        predictions: Optional array or Series containing predicted values.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object showing the time series plot.
    """
    # Extract the specific row's features using the row_id as the index
    location_features = features.loc[row_id]

    # Check if location_features is empty
    if location_features.empty:
        raise ValueError(f"No data found for row ID {row_id}.")

    # Extract the actual target value for the given row_id
    actual_target = targets.loc[row_id]  # Assuming row_id is the index of targets

    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = location_features[time_series_columns].tolist()

    # Generate corresponding timestamps for the time series
    pickup_hour = pd.Timestamp(location_features["pickup_hour"])
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h",
    )

    # Create the plot title with relevant metadata
    title = f"Pickup Hour: {pickup_hour}, Location ID: {location_features['pickup_location_id']}"

    # Create the base line plot
    fig = go.Figure()

    # Add historical ride counts as a line
    fig.add_trace(
        go.Scatter(
            x=time_series_dates,
            y=time_series_values,
            mode="lines+markers",
            name="Historical Ride Counts",
        )
    )

    # Add the actual target value as a green marker
    fig.add_trace(
        go.Scatter(
            x=[time_series_dates[-1]],  # Last timestamp
            y=[actual_target],  # Actual target value
            mode="markers",
            marker=dict(color="green", size=10),
            name="Actual Value",
        )
    )

    # Optionally add the prediction as a red marker
    if predictions is not None:
        if hasattr(predictions, "loc"):  # Check if predictions is a pandas Series/DataFrame
            predicted_value = predictions.loc[row_id]
        else:  # Assume predictions is a NumPy array
            predicted_value = predictions[row_id]

        fig.add_trace(
            go.Scatter(
                x=[time_series_dates[-1]],  # Last timestamp
                y=[predicted_value],  # Predicted value
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name="Prediction",
            )
        )

    # Customize the layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Ride Counts",
        template="plotly_white",
    )

    return fig


def plot_prediction(features: pd.DataFrame, prediction: int):
    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = [
        features[col].iloc[0] for col in time_series_columns
    ] + prediction["predicted_demand"].to_list()

    # Convert pickup_hour Series to single timestamp
    pickup_hour = pd.Timestamp(features["pickup_hour"].iloc[0])

    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h",
    )

    # Create a DataFrame for the historical data
    historical_df = pd.DataFrame(
        {"datetime": time_series_dates, "rides": time_series_values}
    )

    # Create the plot title with relevant metadata
    title = f"Pickup Hour: {pickup_hour}, Location ID: {features['pickup_location_id'].iloc[0]}"

    # Create the base line plot
    fig = px.line(
        historical_df,
        x="datetime",
        y="rides",
        template="plotly_white",
        markers=True,
        title=title,
        labels={"datetime": "Time", "rides": "Ride Counts"},
    )

    # Add prediction point
    fig.add_scatter(
        x=[pickup_hour],  # Last timestamp
        y=prediction["predicted_demand"].to_list(),
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )

    return fig