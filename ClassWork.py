# %% import libraries and functions
# import libraries and functions
from re import X
from venv import create
from imports import *
from functions import *

# %% user inputs
# user inputs

# CSV files containing the raw data
payments_data = "RNA Complete Payments Data.csv"
demographics_data = "RNA Demographics Data File.csv"

# Define the threshold date to filter forward from (in 'YYYY-MM-DD' format)
threshold_date_input = "2018-10-01"
threshold_date = pd.to_datetime(threshold_date_input)

# Define the start of the target year, YEAR 0 (in 'YYYY-MM-DD' format)
target_date_input = "2023-09-17"
target_date = pd.to_datetime(target_date_input)

# Number of years to go forward for equity prediction (e.g. 2)
num_years = 2

# Dollars to consider someone a large donator
# (i.e. above this value = largedonation)
large_donation_threshold = 2000

# %% print user inputs
# print user inputs

display(
    Markdown(
        f"**Threshold Date:** {threshold_date.strftime('%d %b %Y')}\n\nEverything before this date is removed and considered as historical data"
    )
)

display(Markdown(f"**Target Year Start Date:** {target_date.strftime('%d %b %Y')}"))
display(Markdown(f"**Number of years for prediction:** {num_years}"))
display(
    Markdown(
        f"**Large Donation Threshold:** ${large_donation_threshold}\n\nAbove this value is considered a large donation"
    )
)

# %% process base data
# process base data

# extract each donor's first and last gift dates
# merge data
# filter by threshold date
# consider this as base data for further analysis
base_data, column_mappings = process_data(payments_data, threshold_date)

# %% analysis excluding target year
# analysis excluding target year

# process recency and frequency data
recency_frequency = get_rec_freq(base_data, target_date)

# bin recency data
recency_bin_column = "recency_bin"
binned_recency_data, recency_bin_column, bin_labels = make_bins(
    recency_frequency, recency_bin_column
)

# donor equity analysis
donor_equity = targdol_stats(binned_recency_data)

# transition probabilities between recency bins
transition_probabilities = create_transition_matrix(
    binned_recency_data, recency_bin_column
)


# %% analysis including target year

# analysis including target year

target_date_plus_year = target_date + timedelta(days=366)
display(
    Markdown(f"**Target Year End Date:** {target_date_plus_year.strftime('%d %b %Y')}")
)

target_year_recency_frequency = get_rec_freq(base_data, target_date_plus_year)

target_year_binned_recency_data, recency_bin_column, bin_labels = make_bins(
    target_year_recency_frequency,
    recency_bin_column,
)

target_year_transition = create_transition_matrix(
    target_year_binned_recency_data, recency_bin_column
)

predicted_equity = markov_prediction(
    donor_equity,
    transition_probabilities,
    target_year_transition,
    bin_labels,
    target_date,
    num_years,
)

# %% calibrate individual donor probability model
# calibrate individual donor probability model

donor_probabilities = get_decay_parameters(binned_recency_data)

# %% demographic analysis
# demographic analysis

# Merge donor data with demographics data
merged_data = demographic_merge(donor_probabilities, demographics_data)

# Now that merged_data is created, we run the regression model function
X_columns = ["AGE", "D_Male"]  # List of independent variables
y_column = "prob_donate"  # Dependent variable

# Call the function after running demographic_merge
regression_summary = build_regression_model(merged_data, X_columns, y_column)
