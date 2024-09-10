import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


@st.cache_data
def load_data(path):
	df = pd.read_csv(path)
	return df

def plot_3d_scatter(x, y, z, metric, metric_label):
	fig = go.Figure(data=[go.Scatter3d(
		x=x,
		y=y,
		z=z,
		customdata=metric,
		mode='markers',
		marker=dict(
			size=6,
			color=metric,
			colorscale='Viridis',
			opacity=0.8,
			colorbar=dict(
				title=metric_label,
				titleside='top',  # Title on top of the colorbar
				orientation='h',  # Horizontal orientation
				thickness=20,  # Thickness of the colorbar
				len=0.7,  # Length of the colorbar (relative to the plot width)
				x=0.5,  # Center the colorbar horizontally
				y=-0.2,  # Position it below the plot
				xanchor='center',  # Align the colorbar by its center
				yanchor='bottom',  # Align the colorbar by its bottom edge
			)
		),
		hovertemplate = (
			'Longitude: %{x:.5f}<br>' +
			'Latitude: %{y:.5f}<br>' +
			'Altitude: %{z:.0f} m<br>' +
			metric_label + ': %{customdata:.0f}<extra></extra>'
		)
	)])

	# tight layout
	fig.update_layout(
		#width="auto",
		height=700,
		autosize=True,
		margin=dict(
			l=0,
			r=0,
			b=0,
			t=0
		),
		scene=dict(
			camera=dict(
				up=dict(
					x=0,
					y=0,
					z=1 #1
				),
				center=dict(
					x=0,
					y=0,
					z=0
				),
				eye=dict(
					x=0, #-1.25
					y=-1.25, #-1.25
					z=1.25, #1.25
				)
			),
			annotations=[dict(
							showarrow=True,
							x=10.4724,
							y=61.1236,
							z=267.6000,
							text="Lillehammer",
							xanchor="center",
							xshift=10,
							arrowcolor="#ffffff"
						),
						dict(
							showarrow=True,
							x=9.4114,
							y=62.0251,
							z=1169,
							text="Grimsladen",
							xanchor="center",
							xshift=10,
							arrowcolor="#ffffff"
						),
						dict(
							showarrow=True,
							x=8.7514,
							y=61.2044,
							z=1292.8,
							text="Slettefjell",
							xanchor="center",
							xshift=10,
							arrowcolor="#ffffff"
						),
						dict(
							showarrow=True,
							x=7.8126,
							y=61.0471,
							z=405,
							text="Borgund Stavkyrkje",
							xanchor="center",
							xshift=10,
							arrowcolor="#ffffff"
						),
						dict(
							showarrow=True,
							x=7.5015,
							y=60.6025,
							z=1227,
							text="Finse",
							xanchor="center",
							xshift=10,
							arrowcolor="#ffffff"
						)
					],
			xaxis_title='Longitude',
			yaxis_title='Latitude',
			zaxis_title='Altitude',
			dragmode='turntable',
			aspectratio = dict( x=1.5, y=1.5, z=0.2 ),
			aspectmode = 'manual'
		),
		showlegend = False
	)

	return fig



def calculate_normalized_power(df: pd.DataFrame) -> float:
	"""
	Calculate the Normalized Power (NP) from a pandas DataFrame column 'power'.

	Args:
	df (pd.DataFrame): DataFrame containing a column 'power' with power data in watts.

	Returns:
	float: The calculated Normalized Power (NP).
	"""
	
	# Ensure the 'power' column exists
	if 'power' not in df.columns:
		raise ValueError("DataFrame must contain a 'power' column")
	
	# Step 1: Calculate the 30-second rolling average of the 'power' column
	rolling_power = df['power'].rolling(window=30, min_periods=1).mean()

	# Step 2: Raise each rolling power average to the fourth power
	power_raised = np.power(rolling_power, 4)

	# Step 3: Compute the average of the raised values
	avg_power_raised = power_raised.mean()

	# Step 4: Take the fourth root of the average power
	normalized_power = int(np.power(avg_power_raised, 1/4))

	return normalized_power


import pandas as pd

def calculate_time_difference(start_time, end_time):
	"""
	Calculate the time difference between two pandas datetime objects and return in hours and minutes.

	Args:
	start_time (pd.Timestamp): Start datetime
	end_time (pd.Timestamp): End datetime

	Returns:
	str: Time difference in 'hours:minutes' format
	"""
	# Calculate the time difference as a timedelta object
	time_diff = end_time - start_time
	
	# Extract total seconds from the timedelta
	total_seconds = time_diff.total_seconds()
	
	# Convert seconds to hours and minutes
	hours = int(total_seconds // 3600)
	minutes = int((total_seconds % 3600) // 60)
	
	return f"{hours}:{minutes:02d}"  # Return formatted time difference as 'hours:minutes'
