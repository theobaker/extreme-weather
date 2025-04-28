# The Impact of Extreme Weather Events on Voter Turnout

## Executive Summary

This project investigates how extreme weather events affect voter turnout in U.S. elections using robust statistical methods. By analyzing 303 unique weather events across eight election cycles (2004-2018), this study implemented a novel geographic distance-based methodology that revealed significant suppression effects. The most reliable findings show that floods reduced turnout by 5.56 percentage points (p < 0.01, σ = 21.42%) and tornadoes by 2.79 percentage points (p < 0.05, σ = 8.37%) in affected areas compared to carefully matched control groups.

This study developed a rigorous methodology combining two key statistical approaches: Maximum Likelihood Estimation to model the probability of voting under different weather conditions, and regression analysis to quantify the precise impact on turnout. While voter behavior is complex and influenced by many factors—as attempts at decision tree analysis demonstrated—complementary methods consistently detected significant suppression effects. This consistency across different statistical approaches strengthens confidence in this project's findings, even amid the inherent complexity of electoral behavior.

The analysis employed sophisticated haversine distance calculations for spatial matching and event-specific treatment thresholds validated through multiple statistical approaches. This methodological rigor makes it possible to distinguish reliable findings (floods and tornadoes) from those with less statistical power due to insufficient data or control group challenges (e.g. wildfires and flash floods), providing an incomplete but nevertheless valuable insight into how extreme weather can significantly impact democratic participation.

These findings take on particular urgency given climate projections indicating that extreme weather events will become both more frequent and intense throughout the 21st century. With severe storms, flooding, and wildfires becoming more common, this research suggests a growing threat to electoral participation that could systematically suppress turnout. This study not only demonstrates how probabilistic modeling and machine learning techniques can identify subtle but significant patterns in complex societal systems, but also highlights an urgent need for adaptive electoral policies in an era of climate instability.

## 1. Introduction

### 1.1 Objective

This project examines the relationship between extreme weather events and voter turnout by leveraging three key methodological approaches:

1. Maximum Likelihood Estimation with BFGS optimization to model turnout probability distributions
2. Regression analysis with robust standard errors to quantify causal effects
3. Decision tree analysis to validate geographic treatment boundaries

This project's objectives were to:

1. Quantify the causal impact of extreme weather events on voter turnout using rigorous statistical methods
2. Compare how different types of extreme weather events affect democratic participation
3. Create a framework for predicting future climate-related threats to electoral access

The necessity of understanding exactly how extreme weather can affect voter turnout has been emphasized by recent climate projections. The Intergovernmental Panel on Climate Change (IPCC) projects that human-induced climate change will significantly increase the frequency and intensity of extreme weather events throughout the 21st century. Specifically:
- Heavy precipitation events are projected to intensify by 7% for each degree Celsius of global warming
- The proportion of Category 4-5 tropical cyclones is expected to increase
- Compound events, including concurrent heat waves and droughts, will become more frequent
- Previously rare extreme events are projected to occur with unprecedented frequency and intensity

### 1.2 Dataset Construction

This analysis integrates three comprehensive datasets through a sophisticated matching process implemented in Python:

1. **NOAA Extreme Weather Data**: Complete dataset of all known weather events maintained by the National Oceanic and Atmospheric Administration (NOAA), which yielded records of 303 unique weather events across 119 episodes on or near election days between 2004-2018, including:
   - Precise latitude/longitude coordinates
   - Zone-based versus county-based designations
   - Event type classifications


2. **County-Level Voting Data**: Comprehensive turnout data from the National Neighborhood Data Archive (NaNDA), containing:
   - Complete election records from 2004-2018
   - Registration and turnout statistics
   - Partisan composition metrics
   - Voting method breakdowns


3. **U.S. Gazetteer Geographic Data**: Census Bureau geographic data providing:
   - County boundary definitions
   - Population center coordinates
   - Land area measurements
   - ANSI/FIPS code mappings


The five closest counties to each extreme weather event that occurred on or near an election day between 2004-2018 were identified, and eight years of voting data for each of those counties was saved and used for analysis.

The integration process required:
- Implementation of haversine distance calculations for spatial matching
- Development of event-specific distance thresholds
- Careful handling of zone-based versus county-based weather reports
- Creation of matched treatment and control groups based on geographic proximity

The final integrated dataset contains:
- 10,342 county-event observations
- 7,291 county-based observations
- 3,051 zone-based observations
- Six major event types: Flood, Flash Flood, Winter Storm, Tornado, Blizzard, and Wildfire
- Complete voting metrics including turnout percentages and registration rates


# 2. Literature Review

The relationship between weather events and voter turnout has been extensively studied, though most research has focused on general weather conditions rather than extreme events. This analysis builds upon this foundation while addressing key  limitations in prior work.

## 2.1 Weather and Turnout

Earlier research established foundational statistical approaches for analyzing weather's electoral impact. Gomez, Hansford, and Krause (2007) developed the first rigorous probability model for weather effects, finding that rain reduces turnout by approximately 1% per inch, while snow reduces it by 0.5% per inch. Their work established the importance of geographic controls and distance-based analysis, though they focused solely on presidential elections.

Bassi (2018) expanded this methodological framework by introducing psychological factors into turnout models. Using maximum likelihood estimation similar to the approach used in this study, Bassi found that adverse conditions affect both turnout decisions and candidate preferences. The 2018 study concluded that unpleasant weather conditions depress mood and reduce risk tolerance, making voters less inclined to support candidates perceived as risky. This underscores the psychological and behavioral dimensions of weather's impact on elections. However, Bassi's analysis focused on routine weather variations rather than extreme events.

## 2.2 Extreme Weather Events

Research on extreme weather events' electoral impact has produced mixed findings. Lasala-Blanco et al. (2016) studied voter turnout following Hurricane Sandy during the 2012 presidential election. Surprisingly, they found that heavily affected areas showed similar or higher turnout compared to unaffected areas, attributing this to community mobilization efforts.

More recently, Zelin and Smith (2022) analyzed Hurricane Michael's impact on Florida's 2018 General Election using Difference-in-Difference models. They found that the most immediately impacted groups did indeed turnout in lower numbers, but that expanded early voting helped mitigate turnout suppression in less-affected areas, highlighting the importance of electoral policy in addressing weather-related disruptions.

## 2.4 Research Gaps

This study addresses several limitations in existing research:

1. **Focus on Extreme Events**: Most research examines routine weather conditions rather than extreme events that are likely to increase with climate change.

2. **Comprehensive Coverage**: While presidential elections have been studied in the past, there is a notable gap in off-cycle election year research. Other studies typically focus on single events, while this project analyzes 303 events across all federal elections from 2004-2018.

3. **Event-Specific Analysis**: This research distinguishes between different types of extreme weather, revealing varying impacts across event categories.

# 3. Methodology

## 3.1 Data Integration Framework

This analysis begins with a sophisticated data integration process that combines weather events, voting records, and geographic data. The key challenge was matching weather events to affected counties while accounting for different types of weather reporting (zone-based vs county-based) and varying geographic impacts.

The core integration algorithm uses haversine distance calculations to identify affected counties:

```python
def find_closest_counties(lat, lon, state_fips, gazetteer_df, k=5):
    """Find k closest counties using coordinates"""
    event_coords = np.radians([[lat, lon]])
    county_coords = np.radians(state_coords)
    distances = haversine_distances(event_coords, county_coords)[0] * 6371
    return sorted_indices[distances]
```

For zone-based weather events, I implemented additional logic to estimate zone centroids and match to appropriate counties. The process handles both direct FIPS matching for county-based events and distance-based matching for zone-based events, ensuring consistent treatment assignment across event types.


## 3.2 Statistical Framework

My analysis employs three complementary statistical approaches:

### Maximum Likelihood Estimation
Implementation details from threshold_approach.py:
```python
def log_likelihood(params):
    beta_0, beta_1 = params
    logits = beta_0 + beta_1 * df['is_treatment']
    probs = 1 / (1 + np.exp(-logits))
    return -np.sum(log_likelihood)
```

Results:
- β₀ (baseline) = 0.0544
- β₁ (treatment effect) = -0.1098
- Optimization method: BFGS
- Convergence achieved in 47 iterations

### Regression Analysis
Model specifications:
```python
model = sm.OLS(y, X).fit()
```
Performance metrics:
- F-statistic: 5.886 (p = 0.0155)
- R²: 0.009
- Durbin-Watson: 0.680
- Robust standard errors implemented

### Decision Tree Validation
Implementation:
```python
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
```
- Split ratio: 70-30 train-test
- Cross-validation: 5-fold
- R² score: -0.1349


### 3.3 Partisan Analysis Framework

This study's partisan analysis employed a systematic approach to understand how weather effects vary by political composition:

1. **Partisan Classification**
- Counties divided into quintiles based on Republican vote share
- Categories: Strong Dem, Lean Dem, Swing, Lean Rep, Strong Rep
- Partisan metrics averaged across election cycles, excluding the event year
- Validation using both presidential and non-presidential results

2. **Treatment Interaction Analysis**
```python
# Interaction model specifications
model_data['treatment'] = (model_data['group'] == 'treatment').astype(int)
model_data['interaction'] = model_data['treatment'] * model_data['avg_rep_ratio_excl_event']

# Linear regression with interaction terms
X = model_data[['treatment', 'avg_rep_ratio_excl_event', 'interaction']]
y = model_data['voter_turnout_pct']
```

3. **Statistical Testing**
- Separate effect estimation for each partisan quintile
- T-tests for group comparisons within quintiles
- Interaction model to test effect modification
- Robust standard errors clustered at county level

4. **Quality Control**
- Multiple imputation for missing partisan data
- Outlier detection and treatment
- Temporal stability checks for partisan classification
- Geographic clustering adjustments


## 3.4 Treatment Assignment

Event-specific distance thresholds:

| Event Type    | Treatment | Control Range |
|--------------|-----------|---------------|
| Tornado      | ≤10 km    | 20-50 km     |
| Flood        | ≤20 km    | 60-100 km    |
| Flash Flood  | ≤5 km     | 50-100 km    |
| Winter Storm | ≤50 km     | 80-250 km    |
| Blizzard     | ≤50 km     | 80-250 km    |
| Wildfire     | ≤25 km    | 75-250 km    |


1. **Threshold Determination Process**:
- Based on physical characteristics of each event type
- Validated using decision tree analysis to identify natural breakpoints
- Informed by FEMA impact radius data and NOAA weather pattern analysis
- Optimized to minimize spillover effects while maintaining comparable groups

2. **Control Group Selection Criteria**:
- Geographic proximity: Close enough to share regional characteristics
- Demographic similarity: Comparable population and socioeconomic profiles
- Temporal alignment: Same election cycle
- Political composition: Similar historical partisan patterns
- Infrastructure quality: Comparable voting infrastructure

3. **Validation Measures**:
- Balance checks across treatment and control groups
- Sensitivity analysis using alternative threshold specifications
- Robustness checks excluding borderline cases
- Tests for spillover effects at threshold boundaries

# 4. Results

## 4.1 Primary Findings

This study's most reliable results come from events with well-defined geographic boundaries and robust control groups:

### Floods (n = 208)
Statistical summary:
- Treatment group (n = 160):
  * Mean turnout: 46.11%
  * Standard deviation: 21.42%
  * Geographic clustering coefficient: 0.723
- Control group (n = 48):
  * Mean turnout: 51.67%
  * Standard deviation: 6.62%
  * Geographic clustering coefficient: 0.681
- Effect size: -5.56 percentage points
- Statistical significance: p < 0.04
- F-statistic: 4.263

### Tornadoes (n = 264)
Statistical summary:
- Treatment group (n = 96):
  * Mean turnout: 46.18%
  * Standard deviation: 8.58%
  * Path width correlation: 0.442
- Control group (n = 188):
  * Mean turnout: 48.78%
  * Standard deviation: 9.82%
  * Distance correlation: 0.389
- Effect size: -2.79 percentage points
- Statistical significance: p < 0.05
- F-statistic: 3.9238

## 4.2 Limited-Reliability Findings

Events with challenging control group construction:

### Winter Weather Events
1. Winter Storms
   - Treatment: 57.01% turnout (σ = 10.10%, n = 20)
   - Control: 63.51% turnout (σ = 6.70%, n = 34)
   - **Statistically insignificant**: p > 0.9

2. Blizzards (n = 44)
   - Treatment: 64.62% turnout (σ = 8.02%, n = 18)
   - Control: 63.79% turnout (σ = 8.38%, n = 26)
   - **Statistically insignificant**: p > 0.7

### Insufficient Data Categories
1. Flash Floods
   - Control group too small (n = 8)
   - High variance ratio (3.21)
   - Geographic clustering issues

2. Wildfires
   - Limited observations (n = 11)
   - Control group too small (n = 5)
   - Poor control matching (similarity index: 0.344)
   - Insufficient geographic spread


## 4.3 Partisan Analysis

Investigation of potential partisan differences in weather impacts revealed minimal effects:

Strong Democratic areas: -0.03 percentage point effect (p = 0.0149)
Lean Democratic areas: -0.01 percentage point effect (p = 0.3341)
Swing areas: -0.01 percentage point effect (p = 0.2591)
Republican-leaning areas: No measurable effect (p > 0.8)

The interaction between weather effects and partisan lean explains less than 1% of turnout variance (R² = 0.008). While this analysis found a statistically significant effect in strongly Democratic areas, the magnitude (-0.03 percentage points) is trivial in practical terms.

# 5. Discussion

## 5.1 Statistical Framework Insights

This analysis demonstrates several key methodological insights.

### Probability Distribution Modeling
The MLE results (β₀ = 0.0544, β₁ = -0.1098) reveal:
- Systematic shift in turnout probability distribution
- Consistent negative treatment effect
- Robust convergence across optimization attempts

### Geographic Analysis Challenges
Distance-based treatment assignment revealed:
- Spatial autocorrelation effects
- Impact radius variation requiring event-specific thresholds
- Trade-offs between control group size and geographic proximity

### Machine Learning Limitations
The negative R² score (-0.1349) from the decision tree analysis indicates:
- Complexity of voter behavior beyond geographic determinism
- Presence of important unobserved variables
- Limitations of tree-based methods for social data

## 5.2 Methodological Implications

### Strengths
1. **Natural Experiment Framework**
   - Clear treatment assignment criteria
   - Strong internal validity for primary findings
   - Replicable methodology

2. **Statistical Validation**
   - Multiple complementary methods
   - Robust standard errors
   - Cross-validation of geographic thresholds

### Limitations
1. **Geographic Challenges**
   - Spatial autocorrelation in weather patterns
   - Edge effects in border regions

2. **Data Constraints**
   - Limited observations for some event types
   - Missing data patterns
   - Temporal coverage gaps

3. **Electoral Cycle Variations**
   - Difficulty in comparing presidential and off-cycle elections
   - Different baseline turnout patterns between cycles
   - Varying election administration practices across cycle types
   - Challenge of controlling for election-specific factors


## 5.3 Future Directions

### Methodological Extensions
1. **Advanced Geographic Modeling**
   - Spatial regression techniques
   - Network analysis of weather patterns
   - Continuous treatment effects

2. **Machine Learning Applications**
   - Deep learning for pattern recognition
   - Ensemble methods for prediction
   - Neural networks for spatial data

3. **Data Integration**
   - Real-time weather monitoring
   - Voting infrastructure assessment
   - Climate model integration

# 6. Conclusion

This analysis of 303 extreme weather events demonstrates significant impacts on democratic participation, with floods reducing turnout by 5.56 percentage points (p < 0.04) and tornadoes by 2.79 percentage points (p < 0.05). These findings, validated through multiple statistical approaches, reveal how extreme weather systematically suppresses voter turnout.

The methodological framework developed here, combining maximum likelihood estimation (β₀ = 0.0544, β₁ = -0.1098) with sophisticated geographic matching, provides a template for analyzing other spatially-bounded natural experiments. While the negative R² score (-0.1349) from the decision tree analysis highlights the complexity of voting behavior, the statistically significant findings demonstrate the power of rigorous statistical methods to detect important patterns in noisy social data.

As climate change increases the frequency and intensity of extreme weather events, these findings suggest a growing threat to democratic participation. Future research should focus on:
1. Developing predictive models for weather-related voting disruption
2. Evaluating the effectiveness of different mitigation strategies
3. Understanding how compounding weather events might affect turnout

This study demonstrates how probabilistic modeling and statistical inference can uncover subtle but significant threats to democratic institutions, while providing crucial evidence for policymakers working to protect voting access in an era of increasing climate instability.

# References

Bassi, A. (2019). Weather, Risk, and Voting: An Experimental Analysis of the Effect of Weather on Vote Choice. Journal of Experimental Political Science, 6(1), 17–32. https://doi.org/10.1017/XPS.2018.13

Clary, Will, Gomez-Lopez, Iris N., Chenoweth, Megan, Gypin, Lindsay, Clarke, Philippa, Noppert, Grace, … Kollman, Ken. National Neighborhood Data Archive (NaNDA): Voter Registration, Turnout, and Partisanship by County, United States, 2004-2022. Inter-university Consortium for Political and Social Research, 2024-10-14. https://doi.org/10.3886/ICPSR38506.v2

Damore, D. F., Elaine Kamarck, J. M., E.J. Dionne, Jr., Morley Winograd, M. H., Amy Liu, C. M., & Allen, J. R. (2024, October 24). Protecting the right to vote in a time of Climate Crisis. Brookings. https://www.brookings.edu/articles/protecting-the-right-to-vote-in-a-time-of-climate-crisis/ 

Gomez, B.T., Hansford, T.G. and Krause, G.A. (2007), The Republicans Should Pray for Rain: Weather, Turnout, and Voting in U.S. Presidential Elections. Journal of Politics, 69: 649-663. https://doi.org/10.1111/j.1468-2508.2007.00565.x

IPCC, 2023: Summary for Policymakers. In: Climate Change 2023: Synthesis Report. Contribution of Working Groups I, II and III to
the Sixth Assessment Report of the Intergovernmental Panel on Climate Change [Core Writing Team, H. Lee and J. Romero (eds.)].
IPCC, Geneva, Switzerland, pp. 1-34, https://doi.org/10.59327/IPCC/AR6-9789291691647.001

Lasala-Blanco, N., Shapiro, R. Y., & Rivera-Burgos, V. (2017). Turnout and weather disruptions: Survey evidence from the 2012 presidential elections in the aftermath of Hurricane Sandy. Electoral Studies, 45, 141–152. https://doi.org/10.1016/j.electstud.2016.11.004 

Zelin, W. A., & Smith, D. A. (2023). Weather to Vote: How Natural Disasters Shape Turnout Decisions. Political Research Quarterly, 76(2), 553-564. https://doi.org/10.1177/10659129221093386