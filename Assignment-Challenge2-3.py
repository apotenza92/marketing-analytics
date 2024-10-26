# %% Imports and Setup
from imports import *
from functions import *


def main():
    """Main execution function for donor equity analysis"""

    # Setup parameters
    params = {
        "payments_data": "RNA Complete Payments Data.csv",
        "demographics_data": "RNA Demographics Data File.csv",
        "threshold_date": pd.to_datetime("2018-10-01"),
        "target_date": pd.to_datetime("2023-09-17"),
        "large_donation_threshold": 2000,
        "value_column": "avgdonation",
    }

    # Process data and run analysis
    run_donor_equity_analysis(params)


def run_donor_equity_analysis(params):
    """Run complete donor equity analysis"""

    # Prepare data
    data = [
        ("Threshold Date", params["threshold_date"]),
        ("Target Date", params["target_date"]),
        ("Large Donation Threshold", params["large_donation_threshold"]),
        ("Value Column", params["value_column"]),
    ]

    # Format values
    formatted_data = []
    for key, value in data:
        try:
            if isinstance(value, (datetime, pd.Timestamp)):
                value = value.strftime("%d %b %Y")
            elif isinstance(value, str):
                value = datetime.strptime(value, "%Y-%m-%d").strftime("%d %b %Y")
            elif isinstance(value, (int, float)):
                value = f"{value:,.2f}"
        except:
            pass
        formatted_data.append((key, value))

    # Generate and display the table
    headers = ["Analysis Parameters", "Value"]
    table = tabulate(formatted_data, headers=headers, tablefmt="pipe")
    display(Markdown(table))

    # Process base data
    base_data, column_mappings = process_data(
        params["payments_data"], params["threshold_date"]
    )

    # Model Calibration
    display(Markdown("# Model Calibration and Campaign Analysis"))
    model_params = calibrate_model(base_data, params["target_date"])

    # Campaign Analysis
    campaign_metrics = analyze_campaigns(base_data, model_params, params)

    # Demographic Analysis
    display(Markdown("# Demographic Analysis"))
    demo_metrics = analyze_demographics(campaign_metrics["donor_equity"], params)

    # Cross Analysis
    display(Markdown("# Cross-Segment Analysis"))
    cross_metrics = analyze_cross_segments(
        demo_metrics["demo_data"], campaign_metrics["donor_equity"]
    )

    # Export results
    export_results(campaign_metrics, demo_metrics, cross_metrics, params)


def calibrate_model(base_data, target_date):
    """Calibrate decay + learning model"""
    display(Markdown("## Model Calibration (Excluding Target Year)"))

    # Get recency-frequency data excluding target year
    rf_data_excl = get_rec_freq(
        base_data.copy(), target_date, include_target_year=False
    )

    # Calibrate parameters
    parameters = get_decay_parameters(rf_data_excl)
    plot_decay_learning_model(parameters)

    return parameters


def analyze_campaigns(base_data, parameters, params):
    """Analyze campaign-level donor equity"""
    display(Markdown("## Campaign-based Donor Equity Analysis"))

    # Generate data including target year
    rf_data_incl = get_rec_freq(
        base_data.copy(), params["target_date"], include_target_year=True
    )

    # Calculate donor equity
    donor_equity = individual_donor_analysis(
        rf_data_incl,
        parameters,
        params["large_donation_threshold"],
        params["value_column"],
    )

    # Calculate campaign metrics
    campaign_summary = calculate_campaign_metrics(donor_equity)

    # Visualize campaign performance
    plot_campaign_performance(campaign_summary)

    return {"donor_equity": donor_equity, "campaign_summary": campaign_summary}


def calculate_campaign_metrics(donor_equity):
    """Calculate comprehensive campaign metrics"""
    campaign_summary = (
        donor_equity.groupby("firstgiftappeal", observed=True)
        .agg(
            {
                "prob_donate": ["mean", "median", "std"],
                "E_value": ["sum", "mean", "median", "std"],
                "id": "count",
            }
        )
        .round(2)
    )

    campaign_summary.columns = [
        "Avg Prob to Donate",
        "Median Prob to Donate",
        "Std Dev Prob",
        "Total Expected Value ($)",
        "Avg Expected Value ($)",
        "Median Expected Value ($)",
        "Std Dev Expected Value ($)",
        "Number of Donors",
    ]

    # Add derived metrics
    campaign_summary["Value per Donor ($)"] = (
        campaign_summary["Total Expected Value ($)"]
        / campaign_summary["Number of Donors"]
    ).round(2)

    return campaign_summary


def plot_campaign_performance(campaign_summary):
    """Create comprehensive campaign performance visualizations"""
    # Separate Bequest for reference
    non_bequest = campaign_summary[campaign_summary.index != "Bequest Donations"]
    bequest_value = (
        campaign_summary.loc["Bequest Donations", "Total Expected Value ($)"]
        if "Bequest Donations" in campaign_summary.index
        else 0
    )

    # Use the new vibrant color palette
    color_palette = plt.get_cmap("tab10").colors
    colors = [color_palette[i % len(color_palette)] for i in range(len(non_bequest))]

    # Plot 1: Total Expected Value
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    sorted_data = non_bequest["Total Expected Value ($)"].sort_values()
    sorted_data.plot(
        kind="barh",
        ax=ax,
        color=colors,
    )
    ax.set_title(
        "Total Expected Value by Campaign\n"
        + "Total expected donation value for each campaign,\n"
        + "showing relative financial impact across different appeals.\n"
        + "Longer bars indicate more valuable campaigns overall.",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Total Expected Value ($)")

    # Add value labels
    for i, v in enumerate(sorted_data):
        ax.text(v, i, f"${v:,.0f}", va="center")

    plt.tight_layout()
    plt.show()

    # Plot 2: Campaign Performance Matrix
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    scatter = ax.scatter(
        non_bequest["Avg Prob to Donate"],
        non_bequest["Avg Expected Value ($)"],
        s=non_bequest["Number of Donors"] / 20,
        alpha=0.7,
        c=colors[: len(non_bequest)],
    )

    # Add campaign labels
    for idx, row, color in zip(non_bequest.index, non_bequest.itertuples(), colors):
        ax.annotate(
            idx,
            (row._1, row._5),  # Using appropriate indices
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", pad=2),
            color=color,
        )

    ax.set_xlabel("Average Probability to Donate")
    ax.set_ylabel("Average Expected Value ($)")
    ax.set_title(
        "Campaign Performance Matrix\n"
        + "Relationship between donation probability and expected value.\n"
        + "Interpretation:\n"
        + "• Top-right: High-performing campaigns (high probability & value)\n"
        + "• Top-left: High-value but low probability campaigns\n"
        + "• Bottom-right: Reliable but lower value campaigns\n"
        + "• Bottom-left: Low-performing campaigns (low probability & value)\n"
        + "• Larger bubbles = More donors",
        fontsize=12,
        pad=10,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot 3: Donor Distribution
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    sorted_donors = non_bequest["Number of Donors"].sort_values()
    sorted_donors.plot(
        kind="barh",
        ax=ax,
        color=colors,
    )
    ax.set_title(
        "Number of Donors by Campaign\n"
        + "Total number of donors participating in each campaign.\n"
        + "Interpretation:\n"
        + "• Longer bars indicate wider reach\n"
        + "• Compare with total value to identify efficiency\n"
        + "• Useful for identifying popular vs. niche campaigns",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Number of Donors")

    for i, v in enumerate(sorted_donors):
        ax.text(v, i, f"{v:,.0f}", va="center")

    plt.tight_layout()
    plt.show()

    # Plot 4: Value Distribution
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.scatter(
        non_bequest["Std Dev Prob"],
        non_bequest["Std Dev Expected Value ($)"],
        s=non_bequest["Number of Donors"] / 20,
        alpha=0.7,
        c=colors[: len(non_bequest)],
    )

    # Add campaign labels
    for idx, row, color in zip(non_bequest.index, non_bequest.itertuples(), colors):
        ax.annotate(
            idx,
            (row._3, row._7),  # Using appropriate indices
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", pad=2),
            color=color,
        )

    ax.set_title(
        "Campaign Variability Analysis\n"
        + "Shows how consistent or variable campaign performance is.\n"
        + "Interpretation:\n"
        + "• Bottom-left: Most consistent campaigns (low variation)\n"
        + "• Top-right: Most variable campaigns (high risk/reward)\n"
        + "• Bottom-right: Consistent value but variable participation\n"
        + "• Top-left: Consistent participation but variable value\n"
        + "• Larger bubbles indicate more donors",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Standard Deviation of Probability")
    ax.set_ylabel("Standard Deviation of Expected Value ($)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Display summary statistics with interpretations in table
    display(Markdown("### Campaign Performance Summary"))

    summary_stats = {
        "Most Valuable Campaign": [
            non_bequest.index[non_bequest["Total Expected Value ($)"].argmax()],
            "Campaign generating highest total expected donations. Prioritize for resource allocation.",
        ],
        "Highest Average Value": [
            non_bequest.index[non_bequest["Avg Expected Value ($)"].argmax()],
            "Campaign with highest per-donor value. Good for targeted, high-value donor engagement.",
        ],
        "Most Donors": [
            non_bequest.index[non_bequest["Number of Donors"].argmax()],
            "Campaign with widest reach. Effective for building donor base and awareness.",
        ],
        "Highest Probability": [
            non_bequest.index[non_bequest["Avg Prob to Donate"].argmax()],
            "Most reliable campaign for repeat donations. Good for stable revenue planning.",
        ],
        "Total Expected Value (Regular)": [
            f"${non_bequest['Total Expected Value ($)'].sum():,.2f}",
            "Total expected donations excluding bequests. Base revenue projection.",
        ],
        "Bequest Expected Value": [
            f"${bequest_value:,.2f}",
            "Expected value from bequest donations. Important for long-term planning.",
        ],
        "Most Consistent Campaign": [
            non_bequest.index[non_bequest["Std Dev Expected Value ($)"].argmin()],
            "Campaign with lowest variability. Reliable for consistent revenue forecasting.",
        ],
    }

    # Convert to table format with three columns
    table_data = [[k, v[0], v[1]] for k, v in summary_stats.items()]

    display(
        Markdown(
            tabulate(
                table_data,
                headers=["Metric", "Value", "Interpretation"],
                tablefmt="pipe",
            )
        )
    )


def analyze_demographics(donor_equity, params):
    """Analyze donor equity across demographic segments"""
    # Merge and process demographic data
    demo_data = demographic_merge(donor_equity, params["demographics_data"])
    demo_data = process_demographic_data(demo_data)
    # export to excel
    demo_data.to_excel("Exports/demographic_data.xlsx")

    # Create age groups
    demo_data["Age_Group"] = pd.cut(
        demo_data["AGE"],
        bins=[0, 30, 45, 60, 75, 100],
        labels=["Under 30", "30-45", "46-60", "61-75", "Over 75"],
    )

    # Run analyses
    age_metrics = analyze_age_groups(demo_data)
    gender_metrics = analyze_gender(demo_data)
    state_metrics = analyze_states(demo_data)

    return {
        "demo_data": demo_data,
        "age_metrics": age_metrics,
        "gender_metrics": gender_metrics,
        "state_metrics": state_metrics,
    }


def analyze_age_groups(demo_data):
    """Analyze donor equity by age group"""
    display(Markdown("## Age Group Analysis"))

    # Calculate age group metrics
    age_summary = (
        demo_data.groupby("Age_Group", observed=True)
        .agg(
            {
                "prob_donate": ["mean", "median", "std"],
                "E_value": ["sum", "mean", "median", "std"],
                "id": "count",
                "AGE": "mean",
            }
        )
        .round(2)
    )

    # Flatten column names
    age_summary.columns = [
        "Avg Prob to Donate",
        "Median Prob to Donate",
        "Std Dev Prob",
        "Total Expected Value ($)",
        "Avg Expected Value ($)",
        "Median Expected Value ($)",
        "Std Dev Expected Value ($)",
        "Number of Donors",
        "Average Age",
    ]

    # Use the vibrant color palette
    color_palette = plt.get_cmap("tab10").colors
    colors = [color_palette[i % len(color_palette)] for i in range(len(age_summary))]

    # Define a function to rotate x-axis labels
    def rotate_x_labels(ax):
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Plot 1: Total Expected Value
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    age_summary["Total Expected Value ($)"].plot(kind="bar", ax=ax, color=colors)
    ax.set_title(
        "Total Expected Value by Age Group",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Total Expected Value ($)", fontsize=12)

    # Format value labels
    for i, v in enumerate(age_summary["Total Expected Value ($)"]):
        ax.text(i, v * 1.02, f"${v:,.0f}", ha="center", fontsize=10)

    rotate_x_labels(ax)
    plt.tight_layout()
    plt.show()

    # Plot 2: Average and Median Values
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    age_summary[["Avg Expected Value ($)", "Median Expected Value ($)"]].plot(
        kind="bar", ax=ax, color=[colors[0], colors[1]]
    )
    ax.set_title(
        "Average vs Median Expected Value by Age Group",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Expected Value ($)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), fontsize=12)
    rotate_x_labels(ax)
    plt.tight_layout()
    plt.show()

    # Plot 3: Number of Donors
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    age_summary["Number of Donors"].plot(kind="bar", ax=ax, color=colors)
    ax.set_title(
        "Number of Donors by Age Group",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Number of Donors", fontsize=12)

    # Format value labels
    for i, v in enumerate(age_summary["Number of Donors"]):
        ax.text(i, v * 1.02, f"{v:,.0f}", ha="center", fontsize=10)

    rotate_x_labels(ax)
    plt.tight_layout()
    plt.show()

    # Plot 4: Probability Distribution
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    age_summary[["Avg Prob to Donate", "Median Prob to Donate"]].plot(
        kind="bar", ax=ax, color=[colors[0], colors[1]]
    )
    ax.set_title(
        "Donation Probability by Age Group",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), fontsize=12)
    rotate_x_labels(ax)
    plt.tight_layout()
    plt.show()

    # Display summary statistics
    display(Markdown("### Age Group Analysis Summary"))
    display(
        Markdown(
            tabulate(
                age_summary.reset_index(),
                headers="keys",
                tablefmt="pipe",
                showindex=False,
            )
        )
    )

    return age_summary


def analyze_gender(demo_data):
    """Analyze donor equity by gender"""
    display(Markdown("## Gender Analysis"))

    gender_summary = (
        demo_data.groupby("D_Male", observed=True)
        .agg(
            {
                "prob_donate": ["mean", "median", "std"],
                "E_value": ["sum", "mean", "median", "std"],
                "id": "count",
                "AGE": "mean",
            }
        )
        .round(2)
    )

    gender_summary.index = ["Female", "Male"]
    gender_summary.columns = [
        "Avg Prob to Donate",
        "Median Prob to Donate",
        "Std Dev Prob",
        "Total Expected Value ($)",
        "Avg Expected Value ($)",
        "Median Expected Value ($)",
        "Std Dev Expected Value ($)",
        "Number of Donors",
        "Average Age",
    ]

    # Use two vibrant colors from the palette
    color_palette = plt.get_cmap("tab10").colors
    colors = [color_palette[0], color_palette[1]]  # Colors for Female and Male

    # Plot 1: Total Expected Value
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    gender_summary["Total Expected Value ($)"].plot(kind="bar", ax=ax, color=colors)
    ax.set_title(
        "Total Expected Value by Gender\n"
        + "Comparison of total expected donations between genders,\n"
        + "showing relative financial contribution by gender",
        fontsize=12,
        pad=10,
    )
    ax.set_ylabel("Total Expected Value ($)")

    for i, v in enumerate(gender_summary["Total Expected Value ($)"]):
        ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    # Plot 2: Average vs Median Expected Value
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    gender_summary[["Avg Expected Value ($)", "Median Expected Value ($)"]].plot(
        kind="bar", ax=ax, color=colors
    )
    ax.set_title(
        "Average vs Median Expected Value by Gender\n"
        + "Comparison of average and median expected donations,\n"
        + "highlighting potential gender-based donation patterns",
        fontsize=12,
        pad=10,
    )
    ax.set_ylabel("Expected Value ($)")

    plt.tight_layout()
    plt.show()

    # Plot 3: Probability Distribution
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    gender_summary[["Avg Prob to Donate", "Median Prob to Donate"]].plot(
        kind="bar", ax=ax, color=colors
    )
    ax.set_title(
        "Donation Probability by Gender\n"
        + "Average and median probability of donation by gender,\n"
        + "showing likelihood of future donations",
        fontsize=12,
        pad=10,
    )
    ax.set_ylabel("Probability")

    plt.tight_layout()
    plt.show()

    return gender_summary


def analyze_states(demo_data):
    """Analyze donor equity by state"""

    display(Markdown("## State Analysis"))
    # Create state summary
    state_summary = (
        demo_data.groupby("State", observed=True)
        .agg(
            {
                "prob_donate": ["mean", "median", "std"],
                "E_value": ["sum", "mean", "median", "std"],
                "id": "count",
                "AGE": "mean",
            }
        )
        .round(2)
    )

    # Flatten column names
    state_summary.columns = [
        "Avg Prob to Donate",
        "Median Prob to Donate",
        "Std Dev Prob",
        "Total Expected Value ($)",
        "Avg Expected Value ($)",
        "Median Expected Value ($)",
        "Std Dev Expected Value ($)",
        "Number of Donors",
        "Average Age",
    ]

    # Sort states by total expected value
    state_summary = state_summary.sort_values(
        "Total Expected Value ($)", ascending=False
    )

    # Use the new vibrant color palette
    color_palette = plt.get_cmap("tab10").colors
    colors = [color_palette[i % len(color_palette)] for i in range(10)]

    # Plot: Top 10 States by Total Expected Value
    plt.figure(figsize=(15, 8))
    ax1 = plt.gca()
    top_10_states = state_summary.head(10)
    top_10_states["Total Expected Value ($)"].plot(kind="bar", ax=ax1, color=colors)
    ax1.set_title(
        "Top 10 States by Total Expected Value\n"
        + "Comparison of total expected donations from top states,\n"
        + "showing geographical distribution of donor value",
        fontsize=16,
        pad=10,
    )
    ax1.set_ylabel("Total Expected Value ($)", fontsize=12)
    ax1.set_xlabel("")

    for i, v in enumerate(top_10_states["Total Expected Value ($)"]):
        ax1.text(i, v * 1.02, f"${v:,.0f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()

    return state_summary


def analyze_cross_segments(demo_data, donor_equity):
    """Analyze cross-segment patterns"""
    display(Markdown("# Cross-Segment Analysis"))

    # Age and Gender Analysis
    display(Markdown("## Age Group and Gender Analysis"))

    # Use the vibrant color palette
    color_palette = plt.get_cmap("tab10").colors

    # For Gender
    gender_colors = [color_palette[0], color_palette[1]]  # Female, Male

    # For Age Groups
    unique_age_groups = demo_data["Age_Group"].cat.categories
    age_colors = [
        color_palette[i % len(color_palette)] for i in range(len(unique_age_groups))
    ]

    # Plot 1: Total Expected Value by Age and Gender
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    pivot_total = pd.pivot_table(
        demo_data,
        values="E_value",
        index="Age_Group",
        columns="D_Male",
        aggfunc="sum",
        observed=True,
    ).rename(columns={0: "Female", 1: "Male"})

    pivot_total.plot(kind="bar", ax=ax, color=gender_colors)
    ax.set_title(
        "Total Expected Value by Age Group and Gender\n"
        + "Comparison of total expected donations between genders across age groups,\n"
        + "showing relative contribution of each demographic segment",
        fontsize=12,
        pad=10,
    )
    ax.set_ylabel("Total Expected Value ($)")
    ax.legend(title="Gender")
    ax.yaxis.set_major_formatter(lambda x, p: f"${x:,.0f}")

    plt.tight_layout()
    plt.show()

    # Plot 2: Average Expected Value
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    pivot_avg = pd.pivot_table(
        demo_data,
        values="E_value",
        index="Age_Group",
        columns="D_Male",
        aggfunc="mean",
        observed=True,
    ).rename(columns={0: "Female", 1: "Male"})

    pivot_avg.plot(kind="bar", ax=ax, color=gender_colors)
    ax.set_title(
        "Average Expected Value by Age Group and Gender\n"
        + "Average expected donation value for each gender-age combination,\n"
        + "helping identify high-value demographic segments",
        fontsize=12,
        pad=10,
    )
    ax.set_ylabel("Average Expected Value ($)")
    ax.legend(title="Gender")
    ax.yaxis.set_major_formatter(lambda x, p: f"${x:,.0f}")

    plt.tight_layout()
    plt.show()

    # Plot 3: Gender Distribution within Age Groups
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    pivot_donors = pd.pivot_table(
        demo_data,
        values="id",
        index="Age_Group",
        columns="D_Male",
        aggfunc="count",
        observed=True,
    ).rename(columns={0: "Female", 1: "Male"})

    gender_dist = pivot_donors.div(pivot_donors.sum(axis=1), axis=0) * 100
    gender_dist.plot(kind="bar", stacked=True, ax=ax, color=gender_colors)
    ax.set_title(
        "Gender Distribution within Age Groups\n"
        + "Percentage breakdown of gender within each age group,\n"
        + "showing demographic composition of donor segments",
        fontsize=12,
        pad=10,
    )
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Gender")

    plt.tight_layout()
    plt.show()

    # Plot 4: Value per Donor by Age Group and Gender
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    value_per_donor = pivot_total / pivot_donors
    value_per_donor.plot(kind="bar", ax=ax, color=gender_colors)
    ax.set_title(
        "Value per Donor by Age Group and Gender\n"
        + "Average expected value per donor for each demographic segment,\n"
        + "showing efficiency of different donor segments",
        fontsize=12,
        pad=10,
    )
    ax.set_ylabel("Value per Donor ($)")
    ax.legend(title="Gender")
    ax.yaxis.set_major_formatter(lambda x, p: f"${x:,.0f}")

    plt.tight_layout()
    plt.show()

    # Campaign Performance by Demographics
    display(Markdown("## Campaign Performance by Demographics"))

    # Remove Bequest donations for clearer visualization
    non_bequest_data = demo_data[demo_data["firstgiftappeal"] != "Bequest Donations"]

    # Plot 1: Campaign Performance by Age Group
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    campaign_age = pd.pivot_table(
        non_bequest_data,
        values="E_value",
        index="firstgiftappeal",
        columns="Age_Group",
        aggfunc="mean",
        observed=True,
    ).round(2)

    campaign_age.plot(kind="bar", ax=ax, color=age_colors)
    ax.set_title(
        "Average Expected Value by Campaign and Age Group\n"
        + "Comparison of campaign performance across age groups,\n"
        + "helping identify which campaigns resonate with different age segments",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Campaign")
    ax.set_ylabel("Average Expected Value ($)")
    ax.legend(title="Age Group", bbox_to_anchor=(1.05, 1))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(lambda x, p: f"${x:,.0f}")

    plt.tight_layout()
    plt.show()

    # Plot 2: Campaign Performance by Gender
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    campaign_gender = pd.pivot_table(
        non_bequest_data,
        values="E_value",
        index="firstgiftappeal",
        columns="D_Male",
        aggfunc="mean",
        observed=True,
    ).round(2)
    campaign_gender.columns = ["Female", "Male"]

    campaign_gender.plot(kind="bar", ax=ax, color=gender_colors)
    ax.set_title(
        "Average Expected Value by Campaign and Gender\n"
        + "Comparison of campaign performance between genders,\n"
        + "showing gender preferences in campaign response",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Campaign")
    ax.set_ylabel("Average Expected Value ($)")
    ax.legend(title="Gender")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(lambda x, p: f"${x:,.0f}")

    plt.tight_layout()
    plt.show()

    # Display key insights
    display(Markdown("### Cross-Segment Insights"))

    # Calculate key metrics for insights
    top_age_gender_df = pd.pivot_table(
        demo_data,
        values="E_value",
        index=["Age_Group", "D_Male"],
        aggfunc="mean",
        observed=True,
    ).round(2)

    # Update the index to meaningful labels
    top_age_gender_df.index = top_age_gender_df.index.map(
        lambda x: f"{'Male' if x[1] == 1 else 'Female'} {x[0]}"
    )

    top_age_gender = top_age_gender_df["E_value"]

    top_campaign_age = campaign_age.unstack().sort_values(ascending=False)
    top_campaign_gender = campaign_gender.unstack().sort_values(ascending=False)

    # Extract scalar values
    most_valuable_demographic = top_age_gender.idxmax()
    most_valuable_demographic_value = top_age_gender.max()

    # Calculate the campaign with the largest gender gap
    gender_gap = abs(campaign_gender["Male"] - campaign_gender["Female"])
    largest_gender_gap_campaign = gender_gap.idxmax()
    largest_gender_gap_value = gender_gap.max()

    # Format insights with proper number handling
    insights = {
        "Most Valuable Demographic": f"{most_valuable_demographic} (${most_valuable_demographic_value:,.2f})",
        "Top Campaign-Age Combination": (
            f"{top_campaign_age.index[0][0]} - {top_campaign_age.index[0][1]} "
            f"(${top_campaign_age.iloc[0]:,.2f})"
        ),
        "Top Campaign-Gender Combination": (
            f"{top_campaign_gender.index[0][0]} - {top_campaign_gender.index[0][1]} "
            f"(${top_campaign_gender.iloc[0]:,.2f})"
        ),
        "Largest Gender Gap": (
            f"{largest_gender_gap_campaign} " f"(${largest_gender_gap_value:,.2f})"
        ),
        "Most Age-Diverse Campaign": campaign_age.std(axis=1).idxmin(),
    }

    # Define descriptions for each metric
    metric_descriptions = {
        "Most Valuable Demographic": "The demographic group with the highest average expected value.",
        "Top Campaign-Age Combination": "The campaign and age group combination with the highest average expected value.",
        "Top Campaign-Gender Combination": "The campaign and gender combination with the highest average expected value.",
        "Largest Gender Gap": "The campaign with the largest difference in expected value between genders.",
        "Most Age-Diverse Campaign": "The campaign with the smallest variance in expected value across age groups.",
    }

    # Prepare the table rows
    rows = []
    for metric in insights.keys():
        description = metric_descriptions.get(metric, "")
        value = insights[metric]
        rows.append([metric, description, value])

    # Display the table with an additional 'Description' column
    display(
        Markdown(
            tabulate(
                rows,
                headers=["Metric", "Description", "Segment (Value)"],
                tablefmt="pipe",
            )
        )
    )

    return {
        "campaign_age": campaign_age,
        "campaign_gender": campaign_gender,
        "age_gender_metrics": {
            "total_value": pivot_total,
            "avg_value": pivot_avg,
            "donors": pivot_donors,
            "gender_dist": gender_dist,
            "value_per_donor": value_per_donor,
        },
    }


def export_results(campaign_metrics, demo_metrics, cross_metrics, params):
    """Export all analysis results"""
    filename = (
        f"donor_equity_analysis"
        f"_thresh.{params['threshold_date'].strftime('%d-%m-%y')}"
        f"_targ.{params['target_date'].strftime('%d-%m-%y')}"
        f"_large.{params['large_donation_threshold']}.xlsx"
    )

    with pd.ExcelWriter(f"Exports/{filename}") as writer:
        # Summary sheet
        pd.DataFrame(
            {
                "Analysis Parameters": [
                    f"Threshold Date: {params['threshold_date'].strftime('%d %b %Y')}",
                    f"Target Date: {params['target_date'].strftime('%d %b %Y')}",
                    f"Large Donation Threshold: ${params['large_donation_threshold']:,}",
                    f"Value Column: {params['value_column']}",
                ]
            }
        ).to_excel(writer, sheet_name="Analysis Parameters")

        # Campaign Analysis
        campaign_metrics["campaign_summary"].to_excel(
            writer, sheet_name="Campaign Summary"
        )

        # Demographic Analysis
        demo_metrics["age_metrics"].to_excel(writer, sheet_name="Age Analysis")
        demo_metrics["gender_metrics"].to_excel(writer, sheet_name="Gender Analysis")
        demo_metrics["state_metrics"].to_excel(writer, sheet_name="State Analysis")

        # Cross Analysis
        cross_metrics["campaign_age"].to_excel(writer, sheet_name="Campaign by Age")
        cross_metrics["campaign_gender"].to_excel(
            writer, sheet_name="Campaign by Gender"
        )

        # Age-Gender Analysis
        for name, df in cross_metrics["age_gender_metrics"].items():
            sheet_name = f"Age_Gender_{name.replace('_', ' ').title()}"
            df.to_excel(writer, sheet_name=sheet_name)

        # Detailed Analysis
        pd.pivot_table(
            demo_metrics["demo_data"],
            values=["E_value", "prob_donate", "id"],
            index=["firstgiftappeal", "Age_Group"],
            aggfunc={
                "E_value": ["mean", "sum", "std"],
                "prob_donate": ["mean", "std"],
                "id": "count",
            },
            observed=True,
        ).round(2).to_excel(writer, sheet_name="Detailed Cross Analysis")


if __name__ == "__main__":
    main()
    print("Analysis Completed!")
