# %% Imports and Setup
from imports import *
from functions import *


def main():
    """Main execution function for donor analysis"""

    # Setup and load data
    payments_data = "RNA Complete Payments Data.csv"
    threshold_date = pd.to_datetime("2018-10-01")
    target_date = pd.to_datetime("2023-09-17")

    base_data, column_mappings = process_data(payments_data, threshold_date)
    rf_data = get_rec_freq(base_data, target_date, include_target_year=False)

    analyze_and_visualize(
        rf_data, base_data, threshold_date, target_date, column_mappings
    )


def analyze_and_visualize(
    rf_data, base_data, threshold_date, target_date, column_mappings
):
    """Analyze and visualize donor data"""
    non_bequest_data = rf_data[rf_data["firstgiftappeal"] != "Bequest Donations"]
    categories = sorted(non_bequest_data["firstgiftappeal"].unique())

    plot_donor_behavior(non_bequest_data, categories, threshold_date, target_date)
    analyze_loyalty(categories, base_data, column_mappings)
    analyze_growth(categories, base_data, threshold_date, target_date, column_mappings)


def plot_donor_behavior(data, categories, threshold_date, target_date):
    """Plot donor behavior patterns with detailed explanations before the images."""
    # Display main title and description
    display(
        Markdown(
            f"# Donor Behavior Patterns ({threshold_date.strftime('%d %b %Y')} to {target_date.strftime('%d %b %Y')})\n"
        )
    )

    # Add detailed explanation before the plot
    explanation = (
        "## Understanding the Donor Behavior Patterns Graphs\n\n"
        "This series of graphs visualizes the relationship between **donor recency** (how recently a donor made their last donation) "
        "and **donor frequency** (the total number of donations made by a donor) for each campaign category.\n\n"
        "### How Donors are Grouped:\n"
        "- **Grouping by First Gift Appeal**: Each donor is assigned to a single campaign category based on the campaign of their **first donation** (`firstgiftappeal`). "
        "This means that even if a donor has donated to multiple campaigns over time, they are only represented in the graph corresponding to the campaign that first engaged them.\n"
        "- **Unique Representation**: As a result, each donor appears in **only one graph**.\n\n"
        "### Calculating Recency and Frequency:\n"
        "- **Recency**: The number of years since the donor's **most recent donation**, regardless of the campaign. This measures how recently they have engaged with any campaign.\n"
        "- **Frequency**: The **total number of donations** the donor has made across all campaigns. This reflects their overall engagement level.\n\n"
        "### Interpreting the Graphs:\n"
        "- **Scatter Plot**: Each point represents an individual donor, plotted based on their recency and frequency.\n"
        "- **Trendline (Log-Log Scale)**: A trendline is fitted to the data to illustrate the general pattern of donor behavior within each category.\n"
        "- **Slope Interpretation**:\n"
        "  - A **steeper negative slope** suggests that donors are less likely to continue donating as more time passes since their last donation. This indicates that donor engagement decreases sharply over time.\n"
        "  - A **flatter slope** indicates better donor retention and consistent engagement over time, suggesting that donors continue to give regardless of recency.\n\n"
        "### Specific Campaign Insights:\n"
        "- **Example (Red Nose Day)**: For donors whose first donation was to Red Nose Day, the graph shows how these donors behave in terms of their overall giving patterns over time. "
        "A steep negative slope would suggest that these donors tend not to continue donating (to any campaign) as time passes since their last donation.\n"
        "- **Overall Giving Behavior**: The graphs help identify which campaigns are effective at attracting donors who remain engaged over time, not just with the initial campaign but across all giving opportunities.\n\n"
        "Understanding these patterns helps identify which campaigns are more effective at fostering long-term donor engagement and can inform strategies to improve donor retention."
    )
    display(Markdown(explanation))

    # Compute slopes and collect insights first
    insights = []
    for category in categories:
        cat_data = data[data["firstgiftappeal"] == category]
        x = cat_data["recency"] + 0.1  # Avoid log(0) by adding a small value
        y = cat_data["freq"] + 1  # Avoid log(0) by adding 1
        try:
            z = np.polyfit(np.log10(x), np.log10(y), 1)
            slope = z[0]
        except Exception as e:
            print(f"Could not fit trendline for category '{category}': {e}")
            slope = None

        insights.append(
            {
                "Category": category,
                "Slope": slope,
                "Donors_Target": cat_data["targdonate"].sum(),
                "Total_Donors": len(cat_data),
                "Avg_Frequency": cat_data["freq"].mean(),
                "Data": cat_data,
                "Trendline": z if "z" in locals() else None,
            }
        )

    # Remove categories where slope couldn't be calculated
    valid_insights = [insight for insight in insights if insight["Slope"] is not None]

    # Sort categories based on absolute value of slope (flattest first)
    sorted_insights = sorted(valid_insights, key=lambda x: abs(x["Slope"]))

    # Plot each category individually in order of slope
    for insight in sorted_insights:
        category = insight["Category"]
        cat_data = insight["Data"]
        slope = insight["Slope"]
        z = insight["Trendline"]
        x = cat_data["recency"] + 0.1
        y = cat_data["freq"] + 1

        fig, ax = plt.subplots(figsize=(8, 6))

        # Set data points to grey
        ax.scatter(x, y, alpha=0.6, s=30, color="grey")
        try:
            if z is not None:
                x_trend = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
                y_trend = 10 ** np.poly1d(z)(np.log10(x_trend))
                # Set trendline to red
                ax.plot(x_trend, y_trend, "--", color="red", alpha=0.8)

            ax.text(
                0.02,
                0.98,
                f"Slope: {slope:.2f}\nDonors: {len(cat_data):,}",
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )
        except Exception as e:
            print(f"Could not plot trendline for category '{category}': {e}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{category} (Slope: {slope:.2f})", fontsize=14)
        ax.set_xlabel("Recency (Years)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        plt.show()

    # Display insights
    # Prepare insights DataFrame
    insights_df = pd.DataFrame(
        [
            {
                "Category": ins["Category"],
                "Slope": f"{ins['Slope']:.2f}",
                "Donors_Target": f"{ins['Donors_Target']:,}",
                "Total_Donors": f"{ins['Total_Donors']:,}",
                "Avg_Frequency": f"{ins['Avg_Frequency']:.1f}",
            }
            for ins in sorted_insights
        ]
    )

    display(Markdown("## Category Insights (Sorted by Slope)"))

    # Center the text in the table using the 'colalign' parameter
    col_alignments = ("center",) * len(insights_df.columns)

    display(
        Markdown(
            tabulate(
                insights_df,
                headers="keys",
                tablefmt="pipe",
                showindex=False,
                colalign=col_alignments,
            )
        )
    )


def analyze_loyalty(categories, base_data, column_mappings):
    """Analyze donor loyalty metrics with descriptions in markdown before the image."""
    display(Markdown("# Donor Loyalty Analysis"))

    # Add explanatory description before the plot
    description = (
        "This analysis examines the composition of donors within each campaign category, showing the percentage of donors who are single-time donors versus repeat donors.\n\n"
        "**Note:** Repeat donors are counted based on all their historical donations, not just those within the target years. A donor is classified as a repeat donor if they have made more than one donation at any time, provided their last donation was on or after the threshold date.\n\n"
        "This means that even donations made before the threshold date are included when determining whether a donor is a repeat donor. Understanding this helps in assessing overall donor loyalty and engagement over the long term.\n\n"
        "**Interpreting High Percentages:**\n\n"
        "- A **high percentage of repeat donors** in a category suggests that the campaign is effective at fostering ongoing relationships with donors. These campaigns may have strong engagement strategies, effective communication, or causes that resonate deeply with supporters. Investing in such campaigns can lead to sustained funding and a stable donor base.\n"
        "- Conversely, a **high percentage of single-time donors** may indicate that the campaign is more effective at attracting new donors but may struggle to retain them. This could be due to factors like one-time events, lack of follow-up communication, or less engaging content. Identifying these campaigns provides an opportunity to implement strategies aimed at improving donor retention, such as personalized outreach, expressing gratitude, or demonstrating the impact of donations.\n\n"
    )
    display(Markdown(description))

    loyalty_metrics = []
    for category in categories:
        cat_data = base_data[
            base_data[column_mappings["appeal_category_first"]] == category
        ]
        donations_per_donor = cat_data.groupby("id").size()

        total_donors = len(donations_per_donor)
        repeat_donors = sum(donations_per_donor > 1)

        # Calculate median days between donations
        days_between = []
        for _, group in cat_data.groupby("id"):
            if len(group) > 1:
                dates = sorted(group["Giftdate"])
                for i in range(len(dates) - 1):
                    days_between.append((dates[i + 1] - dates[i]).days)

        median_days = np.median(days_between) if days_between else 0

        loyalty_metrics.append(
            {
                "Category": category,
                "Total_Donors": f"{total_donors:,}",
                "Single_Time_Donors": f"{sum(donations_per_donor == 1):,}",
                "Repeat_Donors": f"{repeat_donors:,}",
                "Repeat_Donor_Percentage": f"{(repeat_donors / total_donors * 100):.1f}%",
                "Max_Donations": f"{donations_per_donor.max():,}",
                "Avg_Donations": f"{donations_per_donor.mean():.1f}",
                "Median_Days_Between": f"{median_days:.0f}",
            }
        )

    loyalty_df = pd.DataFrame(loyalty_metrics)
    loyalty_df["Sort_Value"] = (
        loyalty_df["Repeat_Donor_Percentage"].str.rstrip("%").astype(float)
    )
    loyalty_df = loyalty_df.sort_values("Sort_Value", ascending=False).drop(
        "Sort_Value", axis=1
    )

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate and plot percentages
    loyalty_df["Single_Time_Pct"] = (
        loyalty_df["Single_Time_Donors"].str.replace(",", "").astype(float)
        / loyalty_df["Total_Donors"].str.replace(",", "").astype(float)
        * 100
    )
    loyalty_df["Repeat_Donor_Pct"] = (
        loyalty_df["Repeat_Donors"].str.replace(",", "").astype(float)
        / loyalty_df["Total_Donors"].str.replace(",", "").astype(float)
        * 100
    )

    # Define consistent colors
    colors = ["red", "grey"]  # Colors for Single-Time and Repeat Donors

    loyalty_df.plot(
        kind="bar",
        x="Category",
        y=["Single_Time_Pct", "Repeat_Donor_Pct"],
        stacked=True,
        ax=ax,
        title="Donor Composition by Category (% Breakdown)",
        color=colors,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Percentage of Donors")
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.legend(
        ["Single-Time Donors", "Repeat Donors"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Pagebreak
    display(Markdown(r"\pagebreak"))

    # Display metrics
    display(Markdown("## Loyalty Metrics by Category\n(Sorted by Repeat Donor %)"))

    # Create a display DataFrame with shorter column names
    display_df = loyalty_df.drop(["Single_Time_Pct", "Repeat_Donor_Pct"], axis=1).copy()

    # Use tabulate with shorter headers
    table = tabulate(
        display_df,
        headers=[
            "Category",
            "Total",
            "Single",
            "Repeat",
            "Rep %",
            "Max",
            "Avg",
            "Med Days",
        ],
        tablefmt="pipe",
        showindex=False,
    )

    # Add legend below the table
    legend = """
**Legend:**
- Total: Total Donors
- Single: Single Time Donors
- Repeat: Repeat Donors
- Rep %: Repeat Donor Percentage
- Max: Maximum Donations per Donor
- Avg: Average Donations per Donor
- Med Days: Median Days Between Donations
"""

    display(Markdown(table + "\n" + legend))


def analyze_growth(categories, base_data, threshold_date, target_date, column_mappings):
    """Analyze category growth metrics"""
    analysis_start = pd.Timestamp(f"{threshold_date.year + 1}-01-01")

    display(
        Markdown(
            f"# Category Growth Analysis\n"
            f"*Note: Analysis starts from {analysis_start.strftime('%Y')} to ensure complete years. "
            f"Data from {threshold_date.strftime('%b %Y')} to {analysis_start.strftime('%b %Y')} excluded.*\n\n"
            f"*Note: Newsletter shows significant drop during COVID period (2020) which affects overall trends.*"
        )
    )

    # Prepare yearly breakdown
    yearly_breakdown = []
    years = range(analysis_start.year, target_date.year + 1)

    for category in categories:
        cat_data = base_data[
            base_data[column_mappings["appeal_category_first"]] == category
        ]
        for year in years:
            year_data = cat_data[cat_data[column_mappings["gift_date"]].dt.year == year]
            yearly_breakdown.append(
                {
                    "Category": category,
                    "Year": str(year),
                    "Total_Amount": f"${year_data['Amount'].sum():,.2f}",
                    "Num_Donors": f"{year_data['id'].nunique():,}",
                    "Avg_Donation": (
                        f"${year_data['Amount'].mean():.2f}"
                        if len(year_data) > 0
                        else "$0.00"
                    ),
                }
            )

    # Create numeric version for plotting
    yearly_df_numeric = pd.DataFrame(yearly_breakdown)
    yearly_df_numeric["Total_Amount"] = (
        yearly_df_numeric["Total_Amount"]
        .str.replace("$", "")
        .str.replace(",", "")
        .astype(float)
    )
    pivot_df_numeric = yearly_df_numeric.pivot(
        index="Category", columns="Year", values="Total_Amount"
    )

    # Calculate growth metrics
    performance_metrics = []
    for category in categories:
        cat_yearly = yearly_df_numeric[yearly_df_numeric["Category"] == category]
        first_year_amount = float(
            cat_yearly[cat_yearly["Year"] == str(analysis_start.year)][
                "Total_Amount"
            ].iloc[0]
        )
        last_year_amount = float(
            cat_yearly[cat_yearly["Year"] == str(target_date.year)][
                "Total_Amount"
            ].iloc[0]
        )

        total_amount = sum(cat_yearly["Total_Amount"])
        avg_annual = total_amount / len(years)

        performance_metrics.append(
            {
                "Category": category,
                "Total_Amount": f"${total_amount:,.2f}",
                "Avg_Annual_Amount": f"${avg_annual:,.2f}",
                "Overall_Growth": (
                    f"{((last_year_amount / first_year_amount - 1) * 100):+.1f}%"
                    if first_year_amount > 0
                    else "N/A"
                ),
            }
        )

    performance_df = pd.DataFrame(performance_metrics)
    performance_df["Sort_Value"] = (
        performance_df["Total_Amount"]
        .str.replace("$", "")
        .str.replace(",", "")
        .astype(float)
    )
    performance_df = performance_df.sort_values("Sort_Value", ascending=False).drop(
        "Sort_Value", axis=1
    )

    # Define color palette for first visualization
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(len(performance_df))]

    # Create figure for total amounts
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Plot total amounts
    total_amounts = (
        performance_df["Total_Amount"]
        .str.replace("$", "")
        .str.replace(",", "")
        .astype(float)
    )
    ax.bar(
        performance_df["Category"],
        total_amounts,
        color=colors,
    )

    # Customize the plot
    ax.set_title(
        f'Total Donation Amount by Category ({analysis_start.strftime("%Y")} - {target_date.strftime("%Y")})',
        fontsize=14,
        pad=10,
    )
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_formatter(lambda x, p: f"${x:,.0f}")
    ax.set_ylabel("Total Amount ($)")

    plt.tight_layout()
    plt.show()

    # Create figure for trends
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Create a color mapping for categories
    category_colors = dict(zip(performance_df["Category"], colors))

    # Plot trends
    for category in pivot_df_numeric.index:
        yearly_amounts = pivot_df_numeric.loc[category]
        growth = performance_df[performance_df["Category"] == category][
            "Overall_Growth"
        ].iloc[0]
        color = category_colors.get(category, "C0")
        ax.plot(
            years,
            yearly_amounts,
            marker="o",
            label=f"{category} ({growth})",
            linewidth=2,
            markersize=6,
            color=color,
        )

    # Customize the plot
    ax.set_title("Category Performance Trends by Year", fontsize=14, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Donations ($)")
    ax.yaxis.set_major_formatter(lambda x, p: f"${x:,.0f}")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", title="Categories (Overall Growth)"
    )

    plt.tight_layout()
    plt.show()

    # Display tables
    yearly_pivot = yearly_df_numeric.pivot(
        index="Category", columns="Year", values="Total_Amount"
    )
    display(Markdown(r"\pagebreak"))
    display(Markdown("## Year-by-Year Category Performance"))
    display(
        Markdown(
            tabulate(
                yearly_pivot.reset_index(),
                headers="keys",
                tablefmt="pipe",
                showindex=False,
            )
        )
    )

    display(Markdown("## Overall Category Performance Metrics"))
    display(
        Markdown(
            tabulate(performance_df, headers="keys", tablefmt="pipe", showindex=False)
        )
    )


if __name__ == "__main__":
    main()
    print("Analysis Completed!")
