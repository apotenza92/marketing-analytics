from imports import *


# ------ FUNCTIONS FOR DATA SETUP AND RECENCY FREQUENCY ------


def display_table(data, key_header="Key", value_header="Value"):
    """
    Generates and displays a Markdown table from a given dataset, formatting dates and numbers.

    Args:
    - data: Dictionary, list of lists, or iterable containing table data.
    - key_header: Header for the first column (default is "Key").
    - value_header: Header for the second column (default is "Value").
    """

    def format_value(value):
        """Format dates and numbers."""
        # Check if the value is a pandas Timestamp or a standard datetime
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime("%d %b %Y")  # Format as 'DD MMM YYYY'

        # Check if it's a string representing a date
        if isinstance(value, str):
            try:
                date_value = datetime.strptime(value, "%Y-%m-%d")
                return date_value.strftime("%d %b %Y")
            except (ValueError, TypeError):
                pass  # If not a valid date, leave it as is

        # Format numbers with commas and 2 decimal places
        if isinstance(value, (int, float)):
            return f"{value:,.2f}"

        # Return value as is if it's neither a date nor a number
        return value

    # Convert data to list of tuples if it's a dictionary
    if isinstance(data, dict):
        data = list(data.items())

    # Apply formatting to all values in the data
    formatted_data = [(key, format_value(value)) for key, value in data]

    # Generate the table using tabulate
    headers = [key_header, value_header]
    table = tabulate(formatted_data, headers=headers, tablefmt="pipe")

    # Display the table
    display(Markdown(f"{table}"))


def process_data(csv_file, threshold_date):
    """
    This function loads data from a CSV file, processes it by extracting first and last gift dates,
    and filters the data based on a threshold date.

    Returns:
    - A tuple containing:
        1. The filtered DataFrame
        2. A dictionary of column name mappings
    """
    # Create a dictionary to store column name mappings
    column_mappings = {
        "gift_date": "Giftdate",  # original name
        "gift_date_first": "firstgift",  # renamed for first gift
        "gift_date_last": "lastgift",  # renamed for last gift
        "appeal_category": "AppealCat",  # original name
        "appeal_category_first": "FirstGiftAppeal",  # renamed for first gift
    }

    # Load and process the data as before
    df = pd.read_csv(csv_file)

    # Print columns for debugging
    # print("Original columns:", df.columns.tolist())

    df[column_mappings["gift_date"]] = pd.to_datetime(
        df[column_mappings["gift_date"]], dayfirst=True
    )
    df = df.sort_values(by=["id", column_mappings["gift_date"]])

    # Extract first gift information
    df_first = df.groupby("id").first().reset_index()
    df_first = df_first.rename(
        columns={
            column_mappings["gift_date"]: column_mappings["gift_date_first"],
            column_mappings["appeal_category"]: column_mappings[
                "appeal_category_first"
            ],
        }
    )
    df_first = df_first.drop(columns=["Amount", "Giftid", "AppealID", "AppealDesc"])

    # Extract last gift information
    df_last = df.groupby("id").last().reset_index()
    df_last = df_last.rename(
        columns={column_mappings["gift_date"]: column_mappings["gift_date_last"]}
    )
    df_last = df_last.drop(
        columns=[
            "Amount",
            "Giftid",
            "AppealID",
            "AppealDesc",
            column_mappings["appeal_category"],
        ]
    )

    # Merge dataframes
    df_merged = pd.merge(
        df,
        df_first[
            [
                "id",
                column_mappings["gift_date_first"],
                column_mappings["appeal_category_first"],
            ]
        ],
        on="id",
        how="left",
    )
    df_merged = pd.merge(
        df_merged,
        df_last[["id", column_mappings["gift_date_last"]]],
        on="id",
        how="left",
    )

    # Drop the original appeal category column
    df_merged = df_merged.drop(
        columns=[column_mappings["appeal_category"]], errors="ignore"
    )

    # Reset index
    df_merged = df_merged.reset_index(drop=True)
    df_merged.index = df_merged.index + 1

    # Filter by threshold date
    df_filtered = df_merged[
        df_merged[column_mappings["gift_date_last"]] >= threshold_date
    ]
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered.index = df_filtered.index + 1

    # Print final columns for debugging
    # print("Processed columns:", df_filtered.columns.tolist())

    # Create summary dictionary
    summary_dict = {
        "Total Donors": len(df_merged),
        f"Donors forward of {threshold_date.strftime('%d %b %Y')}": len(df_filtered),
    }

    display_table(summary_dict, key_header="Base Data Processing", value_header="Count")

    return df_filtered, column_mappings


def remove_newly_acquired_donors(data, target_date):
    """
    This function removes newly acquired donors in the target year based on the first gift date.
    """
    # input cutoff date variable at top in user inputs
    # Filter the dataframe to remove donors whose first gift is after the cutoff date
    df_filtered = data

    df_filtered = df_filtered[df_filtered["firstgift"] <= target_date]

    # Reset the index after filtering
    df_filtered = df_filtered.reset_index(drop=True)

    # Adjust the index to start at 1
    df_filtered.index = df_filtered.index + 1

    # Create a dictionary with the metric and the count values
    summary_dict = {
        "Removed newly acquired donors": len(data) - len(df_filtered),
        "Remaining donors": len(df_filtered),
    }

    # Display the summary statistics using the display_table function
    display_table(summary_dict, key_header="Removing Target Year", value_header="Count")

    return df_filtered


def get_rec_freq(base_data, target_date, include_target_year=False):
    """
    This function calculates recency, frequency, and avgdonation metrics for each donor based on the target date.
    It also optionally removes newly acquired donors if the target year is not included.

    Args:
    - base_data: DataFrame containing donor data.
    - target_date: Date to calculate recency and frequency from.
    - include_target_year: Boolean flag to include the target year or not.

    Returns:
    - recency_frequency_df: DataFrame containing recency, frequency, and avgdonation metrics for each donor.
    """

    # Adjust the target_date based on the include_target_year flag
    if include_target_year:
        # Add 366 days for leap year consideration
        target_date = target_date + timedelta(days=366)
        year_inclusion_status = "Yes"
    else:
        # Remove newly acquired donors if not including target year
        base_data = remove_newly_acquired_donors(base_data, target_date)
        year_inclusion_status = "No"

    df_filtered = base_data

    # Create an empty list to store the results
    recency_frequency_data = []

    # Group the dataframe by 'id' to process each donor's information
    for donor_id, group in df_filtered.groupby("id"):

        # Sort the group by 'Giftdate'
        group = group.sort_values(by="Giftdate")

        # Initialize variables
        targdonate = 0
        targdol = 0
        freq = 0
        sumdonation = 0
        recency = 9999  # A large initial value for recency
        base_period_donations = 0  # Total donations during the base period
        base_period_count = 0  # Count of donations in the base period

        firstgift = group["firstgift"].iloc[0]  # First gift date for the donor
        lastgift = group["lastgift"].iloc[0]  # Last gift date for the donor
        firstgiftappeal = group["FirstGiftAppeal"].iloc[0]  # First gift appeal

        # Process each row in the donor's group
        for idx, row in group.iterrows():
            amount = row["Amount"]
            giftdate = row["Giftdate"]

            # Calculate the difference (in years) between the target date and the gift date
            diff = (target_date - giftdate).days / 365.25

            # Update the sum of donations
            sumdonation += amount

            # Target period logic
            if diff < 0:  # Donation is in the target period (after the cutoff)
                targdonate = 1
                targdol += amount
            else:
                # Base period logic
                if diff < recency:
                    recency = diff  # Update recency with the smallest difference
                freq += (
                    1  # Increment the frequency for each donation in the base period
                )
                base_period_donations += amount
                base_period_count += 1

        # Calculate avgdonation if there were donations in the base period
        avgdonation = (
            base_period_donations / base_period_count if base_period_count > 0 else 0
        )

        # Append the calculated data to the results list
        recency_frequency_data.append(
            {
                "id": donor_id,
                "targdonate": targdonate,
                "targdol": targdol,
                "recency": recency,
                "freq": freq,
                "firstgiftappeal": firstgiftappeal,
                "sumdonation": sumdonation,
                "avgdonation": avgdonation,
                "lastgift": lastgift,
            }
        )

    # Convert the list of results into a DataFrame
    recency_frequency_df = pd.DataFrame(recency_frequency_data)

    # Create a summary dictionary with individual metrics
    summary_dict = {
        "Target date:": target_date,
        "Target year included?": year_inclusion_status,
        "Donors with recency and frequency data:": len(recency_frequency_df),
    }

    # Display the summary dictionary as a table
    display_table(
        summary_dict, key_header="Recency Frequency Data", value_header="Information"
    )

    return recency_frequency_df


# ------ FUNCTIONS FOR NON PARAMETRIC MARKOV CHAIN MODELS ------


def make_bins(data, bin_column="recency_bin"):
    """
    This function categorizes the 'recency' column into predefined bins and assigns the bin labels.
    The labels follow the pattern: r<1 yr, 1<=r<2 yrs, 2<=r<3 yrs, r>=3 yrs.

    Args:
    - data: DataFrame with a 'recency' column.
    - bin_column: The name of the output column for the recency bins (default is 'recency_bin').

    Returns:
    - DataFrame with added 'recency1' (numeric categories) and a user-defined bin_column (labeled categories) column.
    """

    # Ensure that 'recency' exists in the DataFrame
    if "recency" not in data.columns:
        raise KeyError("'recency' column not found in the DataFrame.")

    # Check for any missing or invalid values in the 'recency' column
    if data["recency"].isnull().any():
        raise ValueError("The 'recency' column contains missing values.")
    if (data["recency"] < 0).any():
        raise ValueError(
            "The 'recency' column contains negative values, which is not expected."
        )

    # Recency categories based on the values in the 'recency' column
    recency_bins = pd.cut(
        data["recency"],
        bins=[-float("inf"), 1, 2, 3, float("inf")],
        labels=["r<1 yr", "1<=r<2", "2<=r<3", "r>=3"],
        right=False,
    )

    # Add the numeric and labeled bin columns
    data["recency1"] = recency_bins.cat.codes  # Numeric codes for the bins
    data[bin_column] = recency_bins  # Labeled bins

    # Get bin labels
    bin_labels = recency_bins.cat.categories

    # Display summary table of bin counts
    bin_counts = data[bin_column].value_counts().sort_index()
    bin_summary = pd.DataFrame({"Recency Bin": bin_labels, "Donor Count": bin_counts})

    # Use tabulate to format the summary as a Markdown table
    summary_table = tabulate(
        bin_summary, headers="keys", tablefmt="pipe", showindex=False
    )

    # Display the table as Markdown
    display(Markdown(summary_table))

    return data, bin_column, bin_labels


def create_transition_matrix(data, bin_column):
    """
    This function creates transition probabilities based on the specified bin_column
    (e.g., 'recency_bin'), and outputs a table where the values and percentages are in separate columns.
    """
    # Create the frequency table for the provided bin_column vs. 'targdonate'
    transition_table = pd.crosstab(data[bin_column], data["targdonate"], margins=True)

    # Calculate percentages for each row
    transition_percentages = transition_table.div(transition_table["All"], axis=0) * 100

    # Create a new DataFrame to store both values and percentages in separate columns
    transition_matrix = pd.DataFrame(index=transition_table.index)

    # Add the raw values (frequencies)
    for column in transition_table.columns:
        transition_matrix[column] = transition_table[column]

    # Add separate columns for the percentages
    for column in transition_table.columns:
        transition_matrix[f"{column}_pct"] = transition_percentages[column].round(2)

    # Rename the index to "Recency Bin" for the Markdown table
    transition_matrix.index.name = "Transition Probabilities for Recency Bin"

    # Convert the final DataFrame to a Markdown table format
    table_output = tabulate(
        transition_matrix, headers="keys", tablefmt="pipe", showindex=True
    )

    # Display the table as Markdown
    display(Markdown(table_output))

    return transition_matrix


def targdol_stats(data):
    """
    This function calculates statistics for each donor and returns them.
    If a collection of datasets is passed in (e.g., multiple data groups), it returns a single table summarizing all groups.
    """
    # Check if the input is a dictionary (multiple data groups)
    if isinstance(data, dict):
        # Initialize an empty dictionary to store equity stats for all groups
        all_equity_stats = {}

        # Loop through each group in the collection
        for group_name, group_data in data.items():
            # Filter the dataframe to include only rows where targdonate is 1
            df_target_donors = group_data[group_data["targdonate"] == 1]

            # Calculate statistics for the 'targdol' column
            stats = (
                df_target_donors["targdol"]
                .agg(["count", "mean", "median", "std", "min", "max", "sum"])
                .round(2)
            )

            # Rename the statistics for better readability
            stats.rename(
                index={
                    "count": "Total number of donors",
                    "mean": "Average donation",
                    "median": "Median donation",
                    "sum": "Total donations",
                    "std": "Standard deviation",
                    "min": "Minimum donation",
                    "max": "Maximum donation",
                },
                inplace=True,
            )

            # Store the stats for this group
            all_equity_stats[group_name] = stats

        # Combine all group stats into a DataFrame
        combined_equity_stats = pd.DataFrame(all_equity_stats).T

        # Create a formatted version for display without modifying the underlying DataFrame
        formatted_equity_stats = combined_equity_stats.copy()

        # Apply dollar formatting for all monetary columns for display only
        monetary_columns = [
            "Average donation",
            "Median donation",
            "Total donations",
            "Standard deviation",
            "Minimum donation",
            "Maximum donation",
        ]
        for col in monetary_columns:
            formatted_equity_stats[col] = formatted_equity_stats[col].apply(
                lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00"
            )

        # Rename the index to "Donor Equity Stats for All Groups" for the Markdown table
        formatted_equity_stats.index.name = "Donor Equity Stats"

        # Convert the formatted DataFrame to a tabulated format
        table_output = tabulate(
            formatted_equity_stats, headers="keys", tablefmt="pipe", showindex=True
        )

        # Display the final table using Markdown
        display(Markdown(f"{table_output}"))

        return combined_equity_stats

    else:
        # Process a single dataset (individual group)
        df_target_donors = data[data["targdonate"] == 1]

        # Calculate statistics for the 'targdol' column
        stats = (
            df_target_donors["targdol"]
            .agg(["count", "mean", "median", "std", "min", "max", "sum"])
            .round(2)
        )

        # Rename the statistics for better readability
        stats.rename(
            index={
                "count": "Total number of donors",
                "mean": "Average donation",
                "median": "Median donation",
                "sum": "Total donations",
                "std": "Standard deviation",
                "min": "Minimum donation",
                "max": "Maximum donation",
            },
            inplace=True,
        )

        # Format the stats for display only
        formatted_stats = stats.copy()
        for label in [
            "Average donation",
            "Median donation",
            "Total donations",
            "Standard deviation",
            "Minimum donation",
            "Maximum donation",
        ]:
            formatted_stats[label] = (
                f"${formatted_stats[label]:,.2f}"
                if pd.notnull(formatted_stats[label])
                else "$0.00"
            )

        # Rename the index for display
        formatted_stats.name = "Donor Equity Stats"

        # Convert stats to a dictionary and display using tabulate
        table_output = tabulate(
            formatted_stats.to_frame().T,
            headers="keys",
            tablefmt="pipe",
            showindex=True,
        )
        display(Markdown(f"{table_output}"))

        return stats


def markov_prediction(
    donor_equity_stats,
    transition_probs,
    target_transition_probs,
    bin_labels,
    target_date,
    num_years,
):
    """
    Predicts donor equity based on the donor transition probabilities and donor statistics.

    Args:
    - donor_equity_stats: Statistics such as median donation values for donors.
    - transition_probs: DataFrame containing the transition probabilities (e.g., '0_pct', '1_pct').
    - bin_labels: List of recency bin labels (e.g., ['r<1 yr', '1<=r<2', '2<=r<3', 'r>=3']).
    - target_date: The start date for predicting donor equity.
    - num_years: The number of years to predict donor equity over.

    Returns:
    - The predicted donor equity for the specified time period.
    """
    display(Markdown("**Predicting donor equity**"))

    # Dynamically create the transition matrix P
    num_bins = len(bin_labels)
    P_dynamic = np.zeros(
        (num_bins, num_bins)
    )  # Square matrix of size [num_bins x num_bins]

    # Populate the transition matrix according to the rules
    for i, label in enumerate(bin_labels):
        if (
            i > 0 and label in transition_probs.index
        ):  # For all bins except the first, move to r<1 yr
            if "1_pct" in transition_probs.columns:
                P_dynamic[i, 0] = transition_probs.loc[label, "1_pct"] / 100

        if i == 0 and bin_labels[0] in transition_probs.index:  # For r<1 yr
            if "1_pct" in transition_probs.columns:
                P_dynamic[0, 0] = (
                    transition_probs.loc[bin_labels[0], "1_pct"] / 100
                )  # Stay in r<1 yr
            if "0_pct" in transition_probs.columns:
                P_dynamic[0, 1] = (
                    transition_probs.loc[bin_labels[0], "0_pct"] / 100
                )  # Move to later bin

        if (
            0 < i < num_bins - 1 and label in transition_probs.index
        ):  # Non-first, non-last bins
            if "0_pct" in transition_probs.columns:
                P_dynamic[i, i + 1] = (
                    transition_probs.loc[label, "0_pct"] / 100
                )  # Move to later bin

    # Last bin handling (r>=3)
    if bin_labels[-1] in transition_probs.index:
        if "0_pct" in transition_probs.columns:
            P_dynamic[-1, -1] = (
                transition_probs.loc[bin_labels[-1], "0_pct"] / 100
            )  # Stay in r>=3
        if "1_pct" in transition_probs.columns:
            P_dynamic[-1, 0] = (
                transition_probs.loc[bin_labels[-1], "1_pct"] / 100
            )  # Move to r<1 yr

    P_dynamic = pd.DataFrame(P_dynamic, index=bin_labels, columns=bin_labels)
    display(Markdown("**Transition Matrix**"))
    display(P_dynamic)

    # Calculate 'n' vector (percentage of donors per bin)
    if not target_transition_probs.empty:
        n = target_transition_probs["All"].iloc[:-1].values  # Exclude 'All' row
    else:
        raise ValueError("Target transition probabilities are empty.")

    display(Markdown(f"**n vector (percentage of donors per bin):**\n\n{n}"))

    # Calculate 'v' (median donation value)
    v = np.zeros(num_bins)
    if np.isscalar(donor_equity_stats):
        v[0] = donor_equity_stats  # If scalar, use it for the first bin
    elif "median" in donor_equity_stats:
        v[0] = donor_equity_stats[
            "median"
        ]  # Use the 'median' value from donor equity stats
    else:
        v[0] = 0  # Default to 0 if 'median' not found
    display(Markdown(f"**v the average/median donation in target year:** {v}\n"))

    # Predict donor equity for the next years
    start_date = target_date + timedelta(days=366) + timedelta(days=1)
    end_date_dynamic = start_date + timedelta(days=365 * num_years - 1)

    display(
        Markdown(
            f"**Start Date = Target Date + 1 year:** {start_date.strftime('%d %b %Y')}"
        )
    )
    display(
        Markdown(
            f"**End Date = End of {num_years} year(s):** {end_date_dynamic.strftime('%d %b %Y')}\n"
        )
    )

    # Calculate donor equity predictions over num_years
    pred = np.dot(np.dot(n.T, P_dynamic), v)
    for year in range(2, num_years + 1):
        pred += np.dot(np.dot(n.T, np.linalg.matrix_power(P_dynamic, year)), v)

    # Format and display the prediction
    formatted_pred = f"${pred:,.2f}"
    display(
        Markdown(
            f"**Predicted Donor Equity from {start_date.strftime('%d %b %Y')} to {end_date_dynamic.strftime('%d %b %Y')}:** {formatted_pred}"
        )
    )

    return pred


# ------ FUNCTIONS FOR PARAMETRIC DECAY + LEARNING MODELS ------


def get_decay_parameters(data):
    """
    Perform optimisation to find the decay + learning model parameters.

    Parameters:
    - data: DataFrame containing the binned or non-binned recency data.

    Returns:
    - Dictionary with the optimised parameters.
    """
    recency_column = "recency1" if "recency1" in data.columns else "recency"

    def loglik(theta):
        k, g0, g1, g2 = theta
        eps = 1e-9  # Small value to avoid log(0)
        pi0 = g0 + g2 * np.exp(-g1 * data["freq"].values)
        ll = np.sum(
            data["targdonate"].values
            * np.log(np.clip(pi0 * k ** data[recency_column].values, eps, None))
            + (1 - data["targdonate"].values)
            * np.log(np.clip(1 - pi0 * k ** data[recency_column].values, eps, None))
        )
        return -ll

    theta0 = np.array([0.5, 0.8, 0.5, -0.5])
    bounds = [(0, 1), (0, 1), (None, None), (None, None)]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = minimize(loglik, theta0, bounds=bounds, method="SLSQP")

        if result.success:
            k, g0, g1, g2 = result.x

            # Create a dictionary to hold the parameters
            parameters = {
                "Decay Model Parameter": ["k", "g0", "g1", "g2"],
                "Value": [k, g0, g1, g2],
            }

            # Convert the dictionary to a DataFrame
            parameters_df = pd.DataFrame(parameters)

            # Convert the DataFrame to a table using tabulate
            table_output = tabulate(
                parameters_df, headers="keys", tablefmt="pipe", showindex=False
            )

            # Display the table using Markdown
            display(Markdown(f"{table_output}"))

            # Return the parameters as a dictionary
            return {"k": k, "g0": g0, "g1": g1, "g2": g2}
        else:
            raise RuntimeError(result.message)

    except RuntimeError as e:
        print(f"Optimisation failed: {str(e)}")
        return None


def plot_decay_learning_model(params):
    """
    Plot the decay + learning model as a function of recency and frequency.

    Parameters:
    - params: Dictionary containing optimized parameters.
    """
    k, g0, g1, g2 = params["k"], params["g0"], params["g1"], params["g2"]

    recency = np.arange(0, 4.05, 0.05)
    y_values = {f: (g0 + g2 * np.exp(-g1 * f)) * (k**recency) for f in range(1, 6)}

    plt.figure(figsize=(10, 6))

    for f, y in y_values.items():
        plt.plot(recency, y, label=f"f={f}")

    plt.xlabel("Recency (years)")
    plt.ylabel("P(donate)")
    plt.title("Decay + Learning Model as a Function of Recency and Frequency")
    plt.legend(title="Frequency")
    plt.grid(True)
    plt.show()


def individual_donor_analysis(
    data, params, large_donation_threshold=None, value_column="avgdonation"
):
    """
    Perform individual donor equity analysis, including optional separation by large and regular donors.

    Parameters:
    - data: DataFrame containing the individual donor data.
    - params: Dictionary with optimized parameters (k, g0, g1, g2).
    - large_donation_threshold: Optional threshold value to classify large donors. If provided, the function will generate separate statistics for large and regular donors.
    - value_column: Column to use for calculating expected donation value (default is 'avgdonation', another option is 'targdol').

    Returns:
    - DataFrame with probabilities, expected values, and optionally the 'largedonation' classification.
    """
    # Unpack the parameters
    k, g0, g1, g2 = params["k"], params["g0"], params["g1"], params["g2"]
    recency_column = "recency1" if "recency1" in data.columns else "recency"

    # Step 1: Calculate probabilities and expected values
    data["prob_donate"] = (g0 + g2 * np.exp(-g1 * data["freq"])) * (
        k ** data[recency_column]
    )
    data["E_value"] = data["prob_donate"] * data[value_column]

    # Step 2: Handle large donation classification if threshold is provided
    if large_donation_threshold is not None:
        data["largedonation"] = data[value_column].apply(
            lambda x: 1 if x >= large_donation_threshold else 0
        )
        large_donors = data[data["largedonation"] == 1]
        regular_donors = data[data["largedonation"] == 0]

        # Generate summary statistics for large donors
        large_donors_summary = (
            large_donors[["prob_donate", "E_value", recency_column, "freq"]]
            .describe()
            .round(2)
        )
        large_donors_summary.loc["sum"] = large_donors[["prob_donate", "E_value"]].sum()
        large_donors_summary.index.name = "Large Donors Statistics"

        # Generate summary statistics for regular donors
        regular_donors_summary = (
            regular_donors[["prob_donate", "E_value", recency_column, "freq"]]
            .describe()
            .round(2)
        )
        regular_donors_summary.loc["sum"] = regular_donors[
            ["prob_donate", "E_value"]
        ].sum()
        regular_donors_summary.index.name = "Regular Donors Statistics"

        # Convert summaries to Markdown tables
        large_table = tabulate(
            large_donors_summary, headers="keys", tablefmt="pipe", showindex=True
        )
        regular_table = tabulate(
            regular_donors_summary, headers="keys", tablefmt="pipe", showindex=True
        )

        # Display the separate tables
        display(
            Markdown(
                f"## Large Donors ({value_column} > ${large_donation_threshold})\n{large_table}"
            )
        )
        display(
            Markdown(
                f"## Regular Donors ({value_column} < ${large_donation_threshold})\n{regular_table}"
            )
        )

    # Step 3: Generate general summary statistics for all donors
    summary = (
        data[["prob_donate", "E_value", recency_column, "freq"]].describe().round(2)
    )
    summary.loc["Sum"] = data[["prob_donate", "E_value"]].sum()

    # Step 4: Extract mean values for the key metrics
    donate_mean = summary.loc["mean", "prob_donate"]
    evalue_mean = summary.loc["mean", "E_value"]

    # Step 5: Create a separate table for Expected donation E_value and Probability of donation
    summary_data = {
        "Summary": [
            f"Expected donation E_value (using {value_column})",
            "Probability of donation (mean)",
        ],
        "Value": [f"${evalue_mean:,.2f}", f"{donate_mean * 100:,.2f}%"],
    }

    # Convert to DataFrame and display as a separate table
    summary_df = pd.DataFrame(summary_data)
    summary_table = tabulate(
        summary_df, headers="keys", tablefmt="pipe", showindex=False
    )
    display(Markdown(f"## Summary"))
    display(Markdown(f"{summary_table}"))

    # Add an index heading to the main summary statistics table
    summary.index.name = f"Summary (using {value_column})"

    # Step 6: Display the main summary statistics as a table
    main_table = tabulate(summary, headers="keys", tablefmt="pipe", showindex=True)
    display(Markdown(f"{main_table}"))

    return data


# ------ FUNCTIONS FOR DEMOGRAPHICS AND REGRESSION MODELS ------


def demographic_merge(recfreq_data, demographic_data_csv):
    """
    This function merges recency and frequency data with demographic data.

    The function identifies an ID column in both datasets, merges the data on the ID, and removes rows with missing values.
    It resets the index of the resulting merged DataFrame.

    Parameters:
    - recfreq_data: DataFrame containing recency and frequency data for individuals.
    - demographic_data_csv: String representing the path to the CSV file containing demographic data.

    Returns:
    - DataFrame: A merged DataFrame combining recency/frequency data and demographic data, with rows containing missing values removed.
    """

    # Helper function to find the ID column in a DataFrame
    def find_id_column(df):
        possible_id_columns = ["id", "ID", "Id", "ConsImpID", "user_id", "UserID"]
        for col in df.columns:
            if col in possible_id_columns:
                return col
        raise KeyError(
            f"No ID column found in the DataFrame. Expected one of: {possible_id_columns}"
        )

    # Load the demographic data from the CSV file
    demographic_data = pd.read_csv(demographic_data_csv)

    # Find the ID column in both recfreq_data and demographic_data
    recfreq_id_col = find_id_column(recfreq_data)
    demographic_id_col = find_id_column(demographic_data)

    display(Markdown(f"Recency and Frequency data rows: {recfreq_data.shape[0]}"))
    display(Markdown(f"Demographic data rows: {demographic_data.shape[0]}"))

    # Merge the recency and frequency data with the demographic data using the detected ID columns
    merged_data = pd.merge(
        recfreq_data,
        demographic_data,
        left_on=recfreq_id_col,
        right_on=demographic_id_col,
        how="left",
    )

    # Delete rows with missing observations
    merged_data.dropna(inplace=True)  # Drops rows with any missing values
    display(
        Markdown(
            f"Rows with missing values removed: {recfreq_data.shape[0] - merged_data.shape[0]}"
        )
    )

    # Reset the index of the merged DataFrame
    merged_data.reset_index(drop=True, inplace=True)
    merged_data.index = merged_data.index + 1

    # print rows in csvs and merged data rows
    display(Markdown(f"Merged data rows: {merged_data.shape[0]}"))

    return merged_data


def process_demographic_data(merged_data):
    """
    This function processes demographic data in a DataFrame by:
    1. Recoding gender into dummy variables ("D_Female" and "D_Male").
    2. Converting the 'DOB' (Date of Birth) column into 'BirthYR' (Birth Year) and calculating the age ('AGE').
    3. Removing rows where the age is less than 18.
    4. Dropping rows with missing or invalid 'DOB' values.

    Parameters:
    - merged_data: DataFrame containing demographic data, including 'Gender' and 'DOB' columns.

    Returns:
    - DataFrame: A new DataFrame with gender recoded into dummies, valid age calculated, and rows with invalid or underage data removed.
    """
    merged_data = (
        merged_data.copy()
    )  # Create a copy of the DataFrame to avoid modifying the original

    # Recode Gender into Dummies
    merged_data["D_Female"] = merged_data["Gender"].apply(
        lambda x: 1 if x == "Female" else 0
    )
    merged_data["D_Male"] = merged_data["Gender"].apply(
        lambda x: 1 if x == "Male" else 0
    )

    # Recode Date of Birth (DOB) into Age
    current_year = datetime.now().year  # Get the current year dynamically
    merged_data["BirthYR"] = pd.to_datetime(
        merged_data["DOB"], dayfirst=True, errors="coerce"
    ).dt.year

    # Handle missing or invalid 'DOB' by dropping rows with NaT in 'BirthYR'
    merged_data = merged_data.dropna(
        subset=["BirthYR"]
    ).copy()  # Ensure we're working with a copy
    display(Markdown(f"Rows with missing 'DOB' values removed: {merged_data.shape[0]}"))

    # Calculate age based on the current year
    merged_data.loc[:, "AGE"] = current_year - merged_data["BirthYR"]

    # Remove rows where Age is less than 18
    merged_data = merged_data[
        merged_data["AGE"] >= 18
    ].copy()  # Ensure we're working with a copy after filtering
    # Display the number of rows removed due to age < 18
    display(Markdown(f"Rows with age less than 18 removed: {merged_data.shape[0]}"))

    # Reset the index of the DataFrame
    merged_data.reset_index(drop=True, inplace=True)
    merged_data.index = merged_data.index + 1

    return merged_data


def make_dummies(df, input_column):
    """
    This function applies one-hot encoding to a specified column in the DataFrame and returns an updated DataFrame with the new dummy columns.
    """
    # Track columns before one-hot encoding
    original_columns = set(df.columns)

    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=[input_column], drop_first=True)

    # Simplify column names by removing the prefix
    prefix = f"{input_column}_"
    df.columns = [
        col.replace(prefix, "") if col.startswith(prefix) else col for col in df.columns
    ]

    # Convert boolean columns to integers
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    # Track columns after one-hot encoding
    output_columns = set(df.columns) - original_columns

    return df, list(output_columns)


def build_regression_model(df, X_columns, y_column):
    """
    Builds a regression model, extracts the top 10 significant coefficients,
    and summarizes key model metrics.

    Args:
    - df: DataFrame containing the data.
    - X_columns: List of columns to be used as independent variables.
    - y_column: The column to be used as the dependent variable.

    Returns:
    - The full coefficient table and a top 10 table of significant coefficients.
    """

    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original

    # List of columns that are missing
    missing_columns = [col for col in X_columns + [y_column] if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in the DataFrame."
        )

    # Extract the X (independent variables) and y (dependent variable)
    X = df[X_columns]
    y = df[y_column]

    # Check for any missing data and drop those rows
    data = pd.concat([X, y], axis=1).dropna()

    # Defragment the DataFrame
    X = data[X_columns]
    y = data[y_column]

    # Add a constant (intercept) to the model
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Get the top summary and display cleanly without row/column numbers
    top_summary = model.summary2().tables[0]

    # Convert the summary table to a DataFrame
    top_summary_df = pd.DataFrame(top_summary)

    # Style the DataFrame: Hide index/columns and left-align the 2nd and 4th columns
    # styled_summary = (
    #     top_summary_df.style.hide(axis="index")
    #     .hide(axis="columns")
    #     .set_properties(
    #         subset=[1, 3], **{"text-align": "left"}
    #     )  # Left-align 2nd and 4th columns
    # )
    # display(styled_summary)

    display(top_summary_df)

    # Get the full summary of coefficients
    full_summary_table = model.summary2().tables[1]

    # Get suburbs from csv file (read the first row as header)
    suburbs_df = pd.read_csv("suburb_list.csv", header=None)  # Load without header

    # Get all suburb names from the first row
    suburbs = suburbs_df.iloc[0, :].tolist()  # Extract first row as a list

    # Filter the rows with suburb names
    suburb_rows = full_summary_table[full_summary_table.index.isin(suburbs)]

    # Filter out the rows with suburb names (non-suburb items) and sort by p-value
    non_suburb_rows = full_summary_table[
        ~full_summary_table.index.isin(suburbs)
    ].sort_values("P>|t|")

    # Sort suburb rows by p-value and get the top suburbs
    top_suburbs = suburb_rows.nsmallest(25, "P>|t|")

    # Display headings and the two tables: all non-suburb items (sorted by p-value) and top suburbs
    display(Markdown("**All Non-Suburb Items (Sorted by P-Value):**"))
    display(non_suburb_rows)  # Show all non-suburb items sorted by p-value

    display(Markdown("**Top Suburbs (Sorted by P-Value):**"))
    display(top_suburbs)  # Show the top suburbs

    return non_suburb_rows, top_suburbs
