import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import sys
import os

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
	sys.path.append(project_root)

from functions import *

#df = load_data("../data/clean/complete.csv") # for local development
df = load_data("data/clean/complete.csv")
df.timestamp = pd.to_datetime(df.timestamp)

days = {
	'Day 1': 11,
	'Day 2': 12,
	'Day 3': 13,
	'Day 4': 14
}

col1, col2 = st.columns([0.8, 0.2])

# Let user select the dynamic metric to display
with st.sidebar:
	st.subheader("Intro")
	st.markdown("""
		Mother North is an unsupported bikepacking race through remote and beautiful landscapes of Norway. 
		Throughout the 1,000km route, riders experienced tough climbs, fast descents, amazing views, reindeer sightings and wind...strong wind.
		This interactive dashboard allows the user to analyse the data my bike computer collected while I completed the ride.
		""")
	st.divider()
	st.subheader("Filters")
	metric_options = ['power', 'cadence', 'heart_rate', 'grade']  # Add other options as needed
	selected_metric = st.selectbox('Select a metric:', metric_options)
	range_options = ['All', 'Day 1', 'Day 2', 'Day 3', 'Day 4']
	#selected_day = st.select_slider('Select a day range:', options=range_options, value=('Day 1', 'Day 4'))
	selected_day = st.radio('Select a day:', options=range_options)

if selected_day == "All":
	pass
else:
	df = df.loc[df.timestamp.dt.day == days[selected_day]]

#df = df.loc[(df.timestamp.dt.day >= days[selected_range[0]]) & (df.timestamp.dt.day <= days[selected_range[1]])]
np_value = calculate_normalized_power(df)

metric_data = df[selected_metric]
metric_label = selected_metric.capitalize()  # This will be used in hover template

# Set the variables for the 3D scatter plot
longitude=df.longitude_deg # x
latitude=df.latitude_deg # y
altitude = df.altitude # z

with col1:
	st.plotly_chart(plot_3d_scatter(longitude, latitude, altitude, metric_data, metric_label))

with col2:
	st.metric(label="Total Distance (Km)", value=int(df.distance_km.max() - df.distance_km.min()), delta=None)
	st.metric(label="Total Elevation (M)", value=int(df.ascent.max() - df.ascent.min()), delta=None)
	st.metric(label="Average Speed (Km/h)", value=int(df.speed_km.mean()), delta=None)
	st.metric(label="Average Heart Rate (Bpm)", value=int(df.heart_rate.mean()), delta=None)
	st.metric(label="Average Power (Watts)", value=int(df.power.mean()), delta=None)
	st.metric(label="Normalized Power (Watts)", value=np_value, delta=None)
	st.metric(label="Average Cadence (Rpm)", value=int(df.cadence.mean()), delta=None)

	# Example usage
	start_time = df.timestamp.min() #pd.Timestamp("2024-09-09 08:30:00")
	end_time = df.timestamp.max() #pd.Timestamp("2024-09-09 15:45:00")

	time_difference = calculate_time_difference(start_time, end_time)

	st.metric(label="Time (HH:MM)", value=time_difference, delta=None)



