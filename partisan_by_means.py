import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

class PartisanWeatherAnalysis:
    """
    Analyzes how weather events affect turnout across different partisan compositions
    """

    def __init__(self, df):
        self.df = df.copy()
        self.results = {}

    def prepare_data(self):
        """Prepare data with partisan metrics and handle missing values"""
        df = self.df.copy()

        # Handle missing values in vote shares
        vote_cols = ['pres_rep_ratio', 'voter_turnout_pct']

        # For presidential vote share, use partisan index if available
        df['rep_lean'] = df['pres_rep_ratio']
        mask = df['rep_lean'].isna() & df['partisan_index_rep'].notna()
        df.loc[mask, 'rep_lean'] = df.loc[mask, 'partisan_index_rep']

        # Create partisan quantiles, handling missing values
        df['partisan_quintile'] = pd.qcut(df['rep_lean'].fillna(df['rep_lean'].mean()), 
                                          q=5, 
                                          labels=['Strong Dem', 'Lean Dem', 'Swing', 'Lean Rep', 'Strong Rep'])

        # Calculate distance ranks and treatment groups
        df['distance_rank'] = df.groupby('EVENT_ID')['DISTANCE_KM'].rank(method='first')
        counties_per_event = df.groupby('EVENT_ID')['distance_rank'].transform('max')

        df['group'] = 'other'
        df.loc[df['distance_rank'] <= (counties_per_event * 0.25), 'group'] = 'treatment'
        df.loc[(df['distance_rank'] > (counties_per_event * 0.25)) & 
               (df['distance_rank'] <= (counties_per_event * 0.5)), 'group'] = 'control'

        self.prepared_df = df
        return df

    def compute_exclusive_average(self):
        """Compute averages excluding the event year."""
        df = self.prepared_df.copy()

        # Ensure we have the necessary columns
        required_columns = ['COUNTY_FIPS', 'VOTING_YEAR', 'EVENT_YEAR', 'pres_rep_ratio']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Group by county and exclude event_year from averages
        df['avg_rep_ratio_excl_event'] = df.apply(
            lambda row: df[(df['COUNTY_FIPS'] == row['COUNTY_FIPS']) & (df['VOTING_YEAR'] != row['EVENT_YEAR'])]['pres_rep_ratio'].mean(),
            axis=1
        )

        self.prepared_df = df
        return df

    def analyze_partisan_representation(self):
        """Analyze whether certain partisan leanings are over/under-represented in affected areas"""
        df = self.prepared_df

        results = {
            'partisan_distribution': {},
            'statistical_tests': {},
            'mean_republican_lean': {}
        }

        # Calculate distributions using non-missing values only
        valid_data = df.dropna(subset=['partisan_quintile', 'group'])

        total_dist = valid_data['partisan_quintile'].value_counts(normalize=True)
        treatment_dist = valid_data[valid_data['group'] == 'treatment']['partisan_quintile'].value_counts(normalize=True)
        control_dist = valid_data[valid_data['group'] == 'control']['partisan_quintile'].value_counts(normalize=True)

        results['partisan_distribution'] = {
            'overall': total_dist.to_dict(),
            'treatment': treatment_dist.to_dict(),
            'control': control_dist.to_dict()
        }

        # Chi-square test on complete cases
        contingency = pd.crosstab(valid_data['partisan_quintile'], valid_data['group'])
        chi2, p_value = stats.chi2_contingency(contingency)[:2]
        results['statistical_tests']['chi_square'] = {
            'statistic': chi2,
            'p_value': p_value,
            'n_observations': len(valid_data)
        }

        # Mean Republican lean by group
        for group in ['treatment', 'control', 'other']:
            group_data = valid_data[valid_data['group'] == group]
            results['mean_republican_lean'][group] = {
                'mean': group_data['avg_rep_ratio_excl_event'].mean(),
                'std': group_data['avg_rep_ratio_excl_event'].std(),
                'n': len(group_data)
            }

        self.results['representation'] = results
        return results

    def analyze_partisan_effects(self):
        """Analyze how weather impacts vary by partisan leaning, using historical mean Republican lean."""
        df = self.prepared_df

        results = {
            'turnout_effects': {},
            'interaction_tests': {},
            'quintile_effects': {}
        }

        # Calculate effects by partisan quintile
        for quintile in df['partisan_quintile'].unique():
            quintile_data = df[df['partisan_quintile'] == quintile]

            # Use only complete cases for this analysis
            valid_data = quintile_data.dropna(subset=['voter_turnout_pct'])

            treatment_turnout = valid_data[valid_data['group'] == 'treatment']['voter_turnout_pct']
            control_turnout = valid_data[valid_data['group'] == 'control']['voter_turnout_pct']

            if len(treatment_turnout) > 0 and len(control_turnout) > 0:
                effect = treatment_turnout.mean() - control_turnout.mean()
                t_stat, p_val = stats.ttest_ind(treatment_turnout, control_turnout)

                results['quintile_effects'][quintile] = {
                    'effect_size': effect,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'n_treatment': len(treatment_turnout),
                    'n_control': len(control_turnout)
                }

        # Interaction analysis with historical mean Republican lean
        model_data = df[df['group'].isin(['treatment', 'control'])].copy()
        model_data['treatment'] = (model_data['group'] == 'treatment').astype(int)
        model_data['interaction'] = model_data['treatment'] * model_data['avg_rep_ratio_excl_event']

        # Keep only complete cases for regression
        required_cols = ['treatment', 'avg_rep_ratio_excl_event', 'interaction', 'voter_turnout_pct']
        valid_model_data = model_data.dropna(subset=required_cols)

        if len(valid_model_data) > 0:
            X = valid_model_data[['treatment', 'avg_rep_ratio_excl_event', 'interaction']]
            y = valid_model_data['voter_turnout_pct']

            model = LinearRegression()
            model.fit(X, y)

            results['interaction_tests']['linear_model'] = {
                'coefficients': dict(zip(['treatment', 'avg_rep_ratio_excl_event', 'interaction'], model.coef_)),
                'r_squared': model.score(X, y),
                'n_observations': len(valid_model_data)
            }

        self.results['partisan_effects'] = results
        return results

    def generate_report(self):
        """Generate comprehensive report of partisan analysis with data quality information"""
        report = []
        report.append("PARTISAN COMPOSITION AND WEATHER EFFECTS ANALYSIS")
        report.append("=============================================")

        # Representation analysis
        if 'representation' in self.results:
            report.append("\n1. PARTISAN REPRESENTATION IN AFFECTED AREAS")
            rep = self.results['representation']

            report.append("\nPartisan Distribution:")
            for group in ['overall', 'treatment', 'control']:
                report.append(f"\n{group.title()} Distribution:")
                for quintile, pct in rep['partisan_distribution'][group].items():
                    report.append(f"  {quintile}: {pct:.1%}")

            chi2_test = rep['statistical_tests']['chi_square']
            report.append(f"\nChi-square test of independence:")
            report.append(f"  Statistic: {chi2_test['statistic']:.2f}")
            report.append(f"  P-value: {chi2_test['p_value']:.4f}")

            report.append("\nMean Republican Lean by Group:")
            for group, stats in rep['mean_republican_lean'].items():
                report.append(f"  {group.title()}: {stats['mean']:.3f} (±{stats['std']:.3f}), n={stats['n']}")

        # Effects analysis
        if 'partisan_effects' in self.results:
            report.append("\n2. DIFFERENTIAL EFFECTS BY PARTISAN LEANING")
            effects = self.results['partisan_effects']

            report.append("\nEffects by Partisan Quintile:")
            for quintile, res in effects['quintile_effects'].items():
                report.append(f"\n{quintile}:")
                report.append(f"  Effect size: {res['effect_size']:.2f} percentage points")
                report.append(f"  P-value: {res['p_value']:.4f}")
                report.append(f"  Sample sizes: {res['n_treatment']} treatment, {res['n_control']} control")

            model = effects['interaction_tests']['linear_model']
            report.append("\nInteraction Model:")
            report.append("  Coefficients:")
            for var, coef in model['coefficients'].items():
                report.append(f"    {var}: {coef:.4f}")
            report.append(f"  R-squared: {model['r_squared']:.3f}")

        return "\n".join(report)

if __name__ == "__main__":
    df = pd.read_csv('/Users/theobaker/Downloads/109final/final_voting_weather_dataset.csv')
    analysis = PartisanWeatherAnalysis(df)
    analysis.prepare_data()
    analysis.compute_exclusive_average()
    analysis.analyze_partisan_representation()
    analysis.analyze_partisan_effects()
    report = analysis.generate_report()
    print(report)
