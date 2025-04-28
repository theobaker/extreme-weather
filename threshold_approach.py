import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

class WeatherImpactAnalysis:
    def __init__(self, df):
        self.df = df.copy()
        self.prepared_df = None

    def prepare_data(self):
        """Prepare the dataset with proper controls and variables."""
        df = self.df.copy()

        # Convert numeric columns
        df['EVENT_YEAR'] = pd.to_numeric(df['EVENT_YEAR'], errors='coerce')
        df['VOTING_YEAR'] = pd.to_numeric(df['VOTING_YEAR'], errors='coerce')
        df['voter_turnout_pct'] = pd.to_numeric(df['voter_turnout_pct'], errors='coerce')
        df['DISTANCE_KM'] = pd.to_numeric(df['DISTANCE_KM'], errors='coerce')

        # Filter for election-year events only
        df = df[df['EVENT_YEAR'] == df['VOTING_YEAR']]

        # Clean and assign groups
        df['EVENT_TYPE'] = df['EVENT_TYPE'].astype(str).str.strip()
        df['group'] = df.apply(self.assign_group, axis=1)

        # Add is_treatment flag safely
        df = df.copy()  # Ensure no SettingWithCopyWarning
        df['is_treatment'] = (df['group'] == 'treatment').astype(int)

        # Drop missing values
        df = df.dropna(subset=['voter_turnout_pct', 'DISTANCE_KM', 'EVENT_TYPE', 'group'])

        self.prepared_df = df
        return df

    def assign_group(self, row):
        """Assign treatment or control group based on distance thresholds."""
        thresholds = self.get_event_specific_threshold(row['EVENT_TYPE'])
        if row['DISTANCE_KM'] <= thresholds['treatment']:
            return 'treatment'
        elif thresholds['control_min'] <= row['DISTANCE_KM'] <= thresholds['control_max']:
            return 'control'
        else:
            return 'excluded'

    def get_event_specific_threshold(self, event_type):
        """Define dynamic distance thresholds based on event type."""
        thresholds = {
            'Tornado': {'treatment': 15, 'control_min': 20, 'control_max': 50},
            'Flash Flood': {'treatment': 5, 'control_min': 80, 'control_max': 200},
            'Flood': {'treatment': 20, 'control_min': 60, 'control_max': 100},
            'Winter Storm': {'treatment': 25, 'control_min': 80, 'control_max': 250},
            'Blizzard': {'treatment': 25, 'control_min': 80, 'control_max': 250},
            'Wildfire': {'treatment': 25, 'control_min': 75, 'control_max': 250},
        }
        return thresholds.get(event_type, {'treatment': 25, 'control_min': 50, 'control_max': 100})

    def mle_analysis(self):
        """Perform MLE to estimate turnout probabilities."""
        df = self.prepared_df[self.prepared_df['group'].isin(['treatment', 'control'])]
        
        # Define log-likelihood function
        def log_likelihood(params):
            beta_0, beta_1 = params
            logits = beta_0 + beta_1 * df['is_treatment']
            probs = 1 / (1 + np.exp(-logits))
            log_likelihood = df['voter_turnout_pct'] * np.log(probs) + (1 - df['voter_turnout_pct']) * np.log(1 - probs)
            return -np.sum(log_likelihood)

        # Optimize log-likelihood
        result = minimize(log_likelihood, [0, 0], method='BFGS')
        params = result.x
        print(f"MLE Results: Beta_0 = {params[0]:.4f}, Beta_1 = {params[1]:.4f}")

    def regression_analysis(self):
        """Run regression analysis for causal inference."""
        df = self.prepared_df[self.prepared_df['group'].isin(['treatment', 'control'])]

        X = sm.add_constant(df['is_treatment'])
        y = df['voter_turnout_pct']

        model = sm.OLS(y, X).fit()
        print(model.summary())

    def decision_tree_analysis(self):
        """Train a decision tree to analyze thresholds and feature importance."""
        df = self.prepared_df[self.prepared_df['group'].isin(['treatment', 'control'])]
        X = df[['DISTANCE_KM', 'is_treatment']]
        y = df['voter_turnout_pct']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        tree.fit(X_train, y_train)

        print(f"Decision Tree R^2 Score: {tree.score(X_test, y_test):.4f}")

    def investigate_all_events(self):
        """Investigate all event types with subgroup analysis."""
        results = {}

        for event_type in self.prepared_df['EVENT_TYPE'].unique():
            df_event = self.prepared_df[self.prepared_df['EVENT_TYPE'] == event_type]
            df_grouped = df_event.groupby('group')['voter_turnout_pct'].agg(['mean', 'median', 'std', 'count'])
            results[event_type] = df_grouped

            treatment = df_event[df_event['group'] == 'treatment']['voter_turnout_pct']
            control = df_event[df_event['group'] == 'control']['voter_turnout_pct']

            # Perform regression for this event type
            X = sm.add_constant(df_event['is_treatment'])
            y = df_event['voter_turnout_pct']
            model = sm.OLS(y, X).fit(cov_type='HC3')  # Robust standard errors

            print(f"Event Type: {event_type}")
            print(df_grouped)
            print(f"Treatment Mean: {treatment.mean():.4f}, Control Mean: {control.mean():.4f}")
            print(f"Treatment Std Dev: {treatment.std():.4f}, Control Std Dev: {control.std():.4f}")
            print(f"P-Value for Treatment Effect: {model.pvalues['is_treatment']:.4f}")
            print(f"F-Statistic: {model.fvalue:.4f}")

# Load data
data_path = '/Users/theobaker/Downloads/109final/final_voting_weather_dataset.csv'
df = pd.read_csv(data_path)

# Initialize analysis object
analysis = WeatherImpactAnalysis(df)
analysis.prepare_data()

# Perform analyses
analysis.mle_analysis()
analysis.regression_analysis()
analysis.decision_tree_analysis()
analysis.investigate_all_events()
