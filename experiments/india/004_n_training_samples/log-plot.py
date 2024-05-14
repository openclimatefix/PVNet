""" Small script to make MAE vs number of batches plot"""

import pandas as df
import plotly.graph_objects as go

data = [[100, 7.779], [300, 7.441], [1000, 7.181], [3000, 7.180], [6711, 7.151]]
df = df.DataFrame(data, columns=["n_samples", "MAE [%]"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["n_samples"], y=df["MAE [%]"], mode="lines+markers"))
fig.update_layout(
    title="MAE % for N samples", xaxis_title="N Samples", yaxis_title="MAE %"
)
# change to log log
fig.update_xaxes(type="log")
fig.show(renderer="browser")
