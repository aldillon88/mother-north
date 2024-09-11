## Mother North Analysis

#### Overview
Mother North is an unsupported 1,000km / 18,000m bikepacking race through remote and beautiful landscapes of Norway. 
This project involved parsing of .fit files from multiple bike GPS computers, cleaning the data and building machine learning models to replace null values with predicted values.
The end result is an interactive dashboard that allows the user to analyse the data my bike computer collected while I completed the Mother North race.

#### Project Setup
##### Python Interpreter
- Python 3.9.6

##### Main Packages Used
- fitparse==1.2.0
- gpxpy==1.6.2
- matplotlib==3.9.2
- notebook==7.2.1
- numpy==2.0.1
- optuna==3.6.1
- pandas==2.2.2
- plotly==5.24.0
- scikit-learn==1.5.1
- scipy==1.13.1
- streamlit==1.38.0
- xgboost==2.1.1

#### Installation
1. Clone this repository:
	1. `git clone https://github.com/aldillon88/mother-north.git`.
2. Create a virtual environment and activate it:
	1. `python -m venv [venv-name]`
	2. `source [venv-name]/bin/activate`.
3. Install the required packages:
	1. `pip install -r requirements.txt`.
4. Run the Streamlit app:
	1. `streamlit run app.py`