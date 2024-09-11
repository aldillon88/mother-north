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

#with st.container():
st.header("Overview")
st.markdown("""
	While not a machine learning app per se, the methods used to create this app relied on machine learning models. 
	The original dataset was sourced from multiple .fit files, all recorded on my Wahoo ELEMNT Roam GPS while competing in the inaugural Mother North bike race in Norway. 
	There is often missing data, for example, when the GPS signal is lost or when a connected sensor occasionally disconnects. 
	This project description will describe the process I followed to handle such missing data before finally displaying the data in this app.
	""")

st.divider()
st.header("Loading the data")
st.markdown("""
	I decided to record sections of the ride each day, as opposed to recording the entire ride in one file. 
	This was mainly because my Wahoo has failed while recording long rides in the past, resulting in lost data. Recording sections at a time helped to avoid that issue.
	However, I accidentally stopped recording about halfway through the final day, which meant I had to fill that section with location data from another rider.
	First step was to import the appropriate libraries and create a combined dataset:
	""")
st.code("""
	import pandas as pd
	import numpy as np
	import fitparse

	# Create a list of file locations
	files = [
		"Mother_North_day_1.fit",
		"Mother_North_day_2a.fit",
		"Mother_North_day_2b.fit",
		"Mother_North_day_3a.fit",
		"Mother_North_day_3b.fit",
		"Mother_North_final_push.fit"
	]

	# Create a function that will parse one .fit file at a time
	def parse_file(file):
		data = []
		fitfile = fitparse.FitFile(file)
		for record in fitfile.get_messages("record"):
			data.append(record.get_values())
		return data

	# Create a loop that will parse all files in the files list
	all_data = []
	for file in files:
		data = parse_file(f"../data/raw/anthony/{file}")
		all_data = all_data + data

	# Create a pandas dataframe
	df = pd.DataFrame(all_data)
	df.sort_values(by="timestamp", inplace=True)
	""")

st.divider()
st.header("Initial cleaning and null handling")
st.markdown("""
	Once the data was loaded into a dataframe, I listed all columns and checked for missing values. I dropped the columns that I was not planning to use and went ahead with filling some of the missing values.
	As you can see in the code below, I dropped rows where the coordinates were exactly the same - i.e. recordings from when I was stationary. I then filled in the small gaps where GPS data was missing.
	I chose to use linear interpolation for this, which fills the gaps between the last non-null value and the next non-null value linearly and with equal increments, as it should provide enough accuracy for this project.
	""")
st.code("""
	# Drop the columns that will not provide any value to the analysis
	cols_to_drop = [
		"calories",
		"battery_soc",
		"left_right_balance",
		"gps_accuracy",
		"enhanced_altitude",
		"enhanced_speed"
	]

	df.drop(columns=cols_to_drop, inplace=True)

	# Fill missing temperature values using bfill and then ffill
	df["temperature"] = df.temperature.bfill().ffill()

	# Drop duplicate rows where the coordinates have not changed - i.e. stationary moments
	df = df.drop_duplicates(subset=["position_lat", "position_long"], keep="first").copy()

	# Fill null values for coordinates and altitude using linear interpolation
	df['position_lat'] = df['position_lat'].interpolate(method='linear')
	df['position_long'] = df['position_long'].interpolate(method='linear')
	df['altitude'] = df['altitude'].interpolate(method='linear')

	# Drop the first row that wasn't filled during interpolation
	df = df.dropna(subset=["position_lat"]).copy()
	""")
st.markdown("""
	At this point the only missing values present were:
	- `grade`: This is the steepness of the terrain at each given location and will be recalculated now that we have filled the missing values for `altitude`.
	- `vertical_speed`: This is measured in meters per second and will also be recalculated.
	- `ascent` and `descent`: These will also be recalculated for the entire dataset.
	- `power`, `heart_rate` and `cadence`: These will be predicted using machine learning models.
	""")

st.divider()
st.header("Appending supplementary data")
st.markdown("""
	Before continuing to fill the missing values, I wanted to append the missing GPS data that I sourced from another rider whom I rode with for a lot of the race. 
	We were riding together for the entire section that I failed to record, so I was confident that the timestamps and associated coordinates and speed were the same.
	""")
st.code("""
	# Parse the other riders .fit file
	file = "../data/raw/supplementary/supplementary.fit"
	fitfile_sup = parse_file(file)
	df_sup = pd.DataFrame(fitfile_sup)

	# Drop duplicate rows where the coordinates have not changed - i.e. stationary moments
	df = df.drop_duplicates(subset=["position_lat", "position_long"], keep="first").copy()

	# Find the last timestamp in the main dataframe.
	last_entry = df.timestamp.max()

	# Create a new dataframe that excludes the unnecessary data and keep only the necessary columns.
	extended_data = df_sup.loc[df_sup.timestamp > last_entry][["timestamp", "position_lat", "position_long", "distance", "altitude", "temperature"]]

	# Concatenate the main dataframe and the extended data
	joined = pd.concat([df, extended_data]).reset_index(drop=True)
	""")
st.markdown("""
	Now that the supplementary data has been appended, we can see the full route represented in the scatter plot below.
	""")

#df = load_data("../data/clean/complete.csv")[["timestamp", "longitude_deg", "latitude_deg"]] # for local development
df = load_data("data/clean/complete.csv")[["timestamp", "longitude_deg", "latitude_deg"]]
df.timestamp = pd.to_datetime(df.timestamp)
cut_off_value = pd.to_datetime("2024-08-14 11:03:02")
df["data_source"] = np.where(df.timestamp <= cut_off_value, 0, 1)
fig = go.Figure(
		data=go.Scatter(
				x=df.longitude_deg,
				y=df.latitude_deg,
				mode='markers',
				marker=dict(
						color=df.data_source
					)
			)
	)

st.plotly_chart(fig)

st.divider()
st.header("Final cleaning and null handling (before using ML)")
st.markdown("""
	In order to recalculate columns such as `speed`, `grade` and `vertical_speed`, the distance between points needs to be known. Coordinates in both .fit files are given in the unit of 'semicircles'.
	We need to convert them to 'degrees', which will allow us to get accurate measurements that take into account the curvature of the earth. 
	For this we use two functions; one to do the conversion to degrees and another to calculate the distance between points with the help of the haversine formula. 
	Once the distance between points is known (`distance_increment`), we can then use cumulative sumation to calculate the total distance traveled at any given point, then finally calculate (or recalculate) the rest of the columns.
	""")
st.code("""
	# Create a function to convert semicircles to degrees
	def semicircles_to_degrees(semicircles):
		return semicircles * (180 / 2**31)

	# Create a function that uses the haversine formula to calculate distance between two lat/lon points
	def haversine(lat1, lon1, lat2, lon2):

		# Earth's radius in meters
		R = 6371000
		
		# Convert latitude and longitude from degrees to radians
		lat1 = np.radians(lat1)
		lon1 = np.radians(lon1)
		lat2 = np.radians(lat2)
		lon2 = np.radians(lon2)
		
		# Differences in coordinates
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		
		# Haversine formula
		a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
		c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
		
		# Distance in meters
		return R * c

	# Convert latitude and longitude from semicircles to degrees
	joined['latitude_deg'] = semicircles_to_degrees(joined['position_lat'])
	joined['longitude_deg'] = semicircles_to_degrees(joined['position_long'])

	# Calculate distance between consecutive points and fill NaN for the first row with 0 (no previous point to calculate distance from)
	joined['distance_increment'] = haversine(joined['latitude_deg'], joined['longitude_deg'], joined['latitude_deg'].shift(), joined['longitude_deg'].shift())
	joined["distance_increment"] = joined["distance_increment"].fillna(0)

	# Calculate cumulative distance
	joined['distance'] = joined['distance_increment'].cumsum()
	""")

st.divider()
st.header("Machine Learning")
st.markdown("""
	This next section will use machine learning to predict heart rate, power and cadence for the rows where such data is missing. In the process of doing so, we will add another column called "pedalling", 
	which will indicate whether I was pedalling or not at any given point. The value of this will become clear in the relevant section below.\n
	I decided to use XGBoost for the regression problems below, as I have had good results using it in the past. XGBoost (Extreme Gradient Boosting) is a powerful, efficient machine learning algorithm based on gradient boosting for decision trees. 
	It is widely used for both regression and classification tasks due to its speed, performance, and ability to handle missing data. 
	XGBoost optimizes model accuracy by minimizing loss using an ensemble of weak learners, while employing techniques like regularization, early stopping, and parallel processing to prevent overfitting and improve efficiency.
	""")

st.subheader("Predicting Heart Rate")
st.markdown("""
	My initial heart rate predictions were based on features from the dataset as it stood without any additional feature engineering. However, the results were fairly poor, so I created the below function 
	in order to add some rolling window features, such as the mean speed, grade and vertical speed values for the trailing 10 rows and the total elevation increment for the same period. 
	The logic behind doing this is based on the idea that it can take time for heart rate to increase or decrease as effort levels change, so understanding what had occurred before any given data point 
	may help the model make better predictions.
	""")
st.code("""
	# Create a function that will add rolling window features to the dataset
	def rolling_features(df, columns, window, agg_function):
		df = df.copy()
		for col in columns:
			col_name = f"rolling_{agg_function}_{window}_{col}"
			df[col_name] = df[col].rolling(window).agg(agg_function).bfill()
		return df

	# Add the rolling window features to the dataset
	df = rolling_features(df, columns=["speed", "grade", "vertical_speed"], window=10, agg_function="mean")
	df = rolling_features(df, columns=["vertical_speed"], window=5, agg_function="mean")
	df = rolling_features(df, columns=["elevation_increment"], window=10, agg_function="sum")
	""")
st.markdown("""
	After checking the correlation values between the target column (`heart_rate`) and the feature columns, I experimented with various feature combinations and hyperparameters, 
	achieving varying degrees of success. Models such as the XGBRegressor from XGBoost provide `feature_importances_` that give further insight into how individual features contribute 
	to the predictions, which also helped to narrow down the final set of features. In addition, some of the correlations between heart rate and some columns appeared to be polynomial, 
	so I experimented with PolynomialFeatures from scikit-learn, which resulted in a better performing model.
	""")
st.code("""
	feature_cols = [
		'distance',
		'altitude',
		'temperature',
		'rolling_mean_10_speed',
		'rolling_mean_10_grade',
		'rolling_mean_10_vertical_speed',
		'rolling_sum_10_elevation_increment'
	]

	training_data = df.drop(columns="timestamp").dropna().copy()
	target = training_data["heart_rate"]
	features = training_data[feature_cols]

	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)

	heart_rate_poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
	heart_rate_poly.fit(X_train)
	X_train_poly = heart_rate_poly.transform(X_train)
	X_test_poly = heart_rate_poly.transform(X_test)

	heart_rate_scaler = StandardScaler()
	heart_rate_scaler.fit(X_train_poly)
	X_train_scaled = heart_rate_scaler.transform(X_train_poly)
	X_test_scaled = heart_rate_scaler.transform(X_test_poly)

	heart_rate_xgb = XGBRegressor(
		learning_rate=0.1, # 0.2
		n_estimators=1600, # 1500
		max_depth=11, # 9
		subsample=0.5, # 0.5
		reg_lambda=20 # 10
	)
	heart_rate_xgb.fit(X_train_scaled, y_train)
	pred = heart_rate_xgb.predict(X_test_scaled)
	""")
st.markdown("""
	In the end, the model outlined in the code above produced the following results:
	- Mean Absolute Error (MAE) of 1.15
	- R2 Score for the training set of 0.998
	- R2 Score for the test set of 0.985\n
	The Max Error was 28.5, which indicates that the model performed poorly when predicting over some outlier values. However, I am happy with the overall performance for this project. 
	So we can move on to the next step, which is predicting cadence.
	""")

st.subheader("Predicting Cadence")
st.markdown("""
	Cadence during cycling can be very irregular due to factors like grade, fatigue, wind etc.. For example, when the grade is steep and negative, a cyclist may not pedal at all and instead allow 
	gravity to get them down the hill. For this reason, I decided to add a new column to the dataset to indicate whether I was pedalling or not. This is straightforward for the rows where cadence 
	and power are known, because if cadence and / or power is above 0 I was clearly pedalling at the time. Using this information, I could then predict whether or not I was pedalling 
	for the rows where cadence and power is unknown. I could then use this to predict cadence only for points where I was likely pedalling.
	""")
st.code("""
	# Create a new categorical columns where 1 means the rider is pedaling, 0 means they are not and -1 means unknown
	df["pedalling"] = np.where(df.cadence > 0, 1, np.where(df.cadence.isna(), -1, 0))

	# Select numerical dtypes
	training_data = df.select_dtypes('number').dropna().copy()
	training_data = training_data.loc[df.pedalling != -1]

	# Split the data into target and features
	target = training_data.pedalling
	features = training_data.drop(columns=["pedalling", "power", "cadence"])

	# Create the train and test sets
	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)

	# Scale the features
	pedalling_scaler = StandardScaler()
	pedalling_scaler.fit(X_train)
	X_train_scaled = pedalling_scaler.transform(X_train)
	X_test_scaled = pedalling_scaler.transform(X_test)

	# Initiate the classification model
	pedalling_clf = LogisticRegression(max_iter=200)
	pedalling_clf.fit(X_train_scaled, y_train)
	pred = pedalling_clf.predict(X_test_scaled)
	""")
st.markdown("""
	The results of this model are below and are satisfactory for this project, so we can now move on to predicting the actual cadence values:
	- Accuracy Score: 0.923
	- Precision Score: 0.934
	- Recall Score: 0.983\n
	After taking a closer look at the distribution of cadence values, I realised that some of them were clearly wrong. This can happen when sensors malfunction. For this reason I decided 
	to replace the outliers with `np.nan` so they can be predicted again once the model is trained.
	""")
st.code("""
	# Create a function that returns outliers for any given numerical column
	def locate_outliers(df, col):
		Q1 = df[col].quantile(0.25)
		Q3 = df[col].quantile(0.75)
		IQR = Q3 - Q1
		upper_bound = Q3 + IQR * 1.5
		lower_bound = Q1 - IQR * 1.5
		upper_outliers = df.loc[df[col] > upper_bound]
		lower_outliers = df.loc[df[col] < lower_bound]
		return upper_bound, lower_bound, upper_outliers, lower_outliers

	# Filter the dataframe by 'pedalling'
	pedalling_subset = df.loc[df.pedalling == 1].dropna(subset="cadence").copy()

	# Calculate outliers and create a list of their respective indexes
	upper_bound, lower_bound, upper_outliers, lower_outliers = locate_outliers(pedalling_subset, "cadence")
	indexes_to_modify = lower_outliers.index.append(upper_outliers.index)

	# Replace values in the 'cadence' column with NaN at the given indexes
	df.loc[indexes_to_modify, 'cadence'] = np.nan
	""")
st.markdown("""
	Even with the outliers removed, cadence numbers can be erratic, so I decided to train the model on smoothed cadence data instead. The scores for this model are as follows:
	- Mean Absolute Error: 2.355
	- R2 Score: 0.943
	""")
st.code("""
	# Prepare training data for cadence predictions
	training_data = df.loc[df.pedalling == 1].dropna().copy()
	training_data["cadence_smoothed"] = training_data.cadence.rolling(10).mean().dropna()
	training_data = training_data.dropna().copy()

	feature_cols = [
		'distance',
		'altitude',
		'speed',
		'grade',
		'vertical_speed',
		'heart_rate',
		'temperature',
		'rolling_mean_10_speed',
		'rolling_mean_10_grade',
		'rolling_mean_10_vertical_speed',
		'rolling_mean_5_vertical_speed',
		'rolling_10_heart_rate'
	]

	target = training_data["cadence_smoothed"]
	features = training_data[feature_cols]

	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)

	cadence_scaler = StandardScaler()
	cadence_scaler.fit(X_train)
	X_train_scaled = cadence_scaler.transform(X_train)
	X_test_scaled = cadence_scaler.transform(X_test)

	cadence_xgb = XGBRegressor(
		eval_metric='mae',
		n_estimators=1700,
		max_depth=16,
		learning_rate=0.1,
		subsample=0.6,
		reg_lambda=55,
		reg_alpha=20
	)
	cadence_xgb.fit(X_train_scaled, y_train)
	cadence_pred = cadence_xgb.predict(X_test_scaled)
	""")

st.subheader("Predicting Power")
st.markdown("""
	Finally, with `heart_rate` and `cadence` predicted, we can predict the final column - `power`. 
	Once again, you can see that I used XGBRegressor for this task and once again decided to train the model on smoothed data. 
	The model below achieved the following scores:
	- Mean Absolute Error: 6.578
	- R2 Score: 0.951
	""")
st.code("""
	# Prepare training data for cadence predictions
	training_data = df.loc[df.pedalling == 1].dropna().copy()
	training_data["power_smoothed"] = training_data.power.rolling(10).mean()

	training_data = training_data.dropna().copy()

	feature_cols = [
		'distance',
		'altitude',
		'speed',
		'grade',
		'vertical_speed',
		'cadence',
		'heart_rate',
		'rolling_mean_10_speed',
		'rolling_mean_10_grade',
		'rolling_10_heart_rate'
	]

	#training_data = df.loc[df.pedalling == 1].dropna()
	target = training_data["power_smoothed"]
	features = training_data[feature_cols]

	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)

	power_scaler = StandardScaler()
	power_scaler.fit(X_train)
	X_train_scaled = power_scaler.transform(X_train)
	X_test_scaled = power_scaler.transform(X_test)

	power_xgb = XGBRegressor(
		eval_metric='mae',
		n_estimators=1700,
		max_depth=15,
		learning_rate=0.07,
		subsample=0.6,
		reg_lambda=55, #L2
		reg_alpha=20 #L1
	)
	power_xgb.fit(X_train_scaled, y_train)
	power_pred = power_xgb.predict(X_test_scaled)
	""")
st.markdown("""
	Now with a full dataset, I was able to build the dashboard in this Streamlit app to analyse the race. The final dataframe could also be exported as a .gpx file and uploaded to platforms 
	like Strava.
	""")
st.divider()
st.header("Final Thoughts")
st.markdown("""
	This project description does not include all of the code written to complete the project. However, it does include the most important snippets to give an indication of the process 
	that I followed. There was a lot of trial and error as I tried to build more performant models, some of which could probably have been avoided if I had spent more time cleaning and exploring 
	the data prior to building the models.\n
	Most of the rows were separated by 1 second time increments, with the occasional larger gap due to rest periods during the race or disconnected sensors / GPS. If I were to do this project again, 
	I would experiement with resampling the data to equal but larger time increments, for example, 3 or 5 seconds using the mean value across that window. This might smooth the data and 
	make it easier to work with when building the ML models.
	""")

















