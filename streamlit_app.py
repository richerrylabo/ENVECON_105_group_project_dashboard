import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    geom_smooth,
    facet_wrap,
    facet_grid,
    labs,
    theme,
    element_text,
    scale_colour_manual,
    coord_cartesian,
    geom_tile,
    scale_fill_gradient,
    scale_x_continuous,
    scale_y_discrete
)




# load the two dashboards’ datasets
df_individual = pd.read_csv("wrangled_data_individual.csv")  # USA
df_group = pd.read_csv("wrangled_data_group.csv")            # Norway

# helper function to apply consistent theme styling
def my_theme(ax, fig, caption=''):
        ax.tick_params(axis='x', labelsize=12)  # set x axis tick font size
        ax.tick_params(axis='y', labelsize=12)  # set y axis tick font size
        ax.xaxis.label.set_size(12)             # set x axis label font size
        ax.yaxis.label.set_size(12)             # set y axis label font size
        ax.title.set_size(16)                   # set title font size
        fig.text(0.8, -0.03, caption,           # add caption below plot
                ha='center', va='bottom', fontsize=12)
        ax.set_facecolor('white')               # set plot background
        fig.patch.set_facecolor('white')        # set figure background
        ax.grid(True, color='black', linewidth=0.5, alpha=0.3)  # add grid
# ensure numeric year for grouping/plotting
for _df in (df_individual, df_group):
    if "Year" in _df.columns:
        _df["Year"] = pd.to_numeric(_df["Year"], errors="coerce")



#Create tabs at the top with linked markdowns to headers


tab1, tab2 = st.tabs(["Individual Project: USA", "Group Project: Norway"])

with tab1:
    st.markdown("Jump to: [Visuals](#visuals) | [Analysis](#analysis) | [Summary Plot](#summary-plot)")
    st.header("Intro")
    st.write(
        """The following visuals and statistics present the results of part an individual case study examining the United States’ role in global carbon dioxide (CO₂) emissions and their potential relationship with climate impacts. 
        CO₂ is the most abundant greenhouse gas emitted by human activities in the U.S. 
        By analyzing emissions data alongside long-term records of global temperature changes and U.S. natural disaster events, this study provides insight into how emissions trends have evolved and how they may connect to climate-related risks.
        Using data wrangling, visualization, and basic statistical analysis, this case study highlights patterns that can inform how we understand the climate consequences of greenhouse gas emissions.  \n\n"""
        "**The case study had two main research questions:**  \n"
        "**1)** How have global CO2 emission rates changed over time? In particular for the US, and how does the US compare to other countries?  \n"
        "**2)** Are CO2 emissions in the US, global temperatures, and natural disaster rates in the US associated?"
    )



    #----------------------------------------
    #----------------------------------------
    #Vis header
    st.header("Visuals")


    #------------------------------------------------------
    #WORLD CO2 EMISSION PER YEAR GRAPH
    #------------------------------------------------------

    #text title
    st.subheader("World CO2:")
    #-------------------------------------------


    # aggregate world CO2 emissions by year
    world = (
        df_individual[df_individual['Indicator'] == 'CO2 Emissions (Metric Tons)']
        .groupby('Year', as_index=False)['Value'].sum()
        .rename(columns={'Value': 'Emissions'})
    )

    # create line plot of world emissions
    fig, ax = plt.subplots()
    ax.plot(world['Year'], world['Emissions'], linewidth=1.5)
    ax.set_title("World CO$_2$ Emissions per Year (1751-2014)")
    ax.set_ylabel("Emissions (Metric Tonnes)")
    ax.set_xlabel("Year")

    # apply theme and add caption
    my_theme(ax, fig, "Limited to reporting countries")

    # show plot in streamlit
    st.pyplot(fig)


    #------------------------------------------------------
    #COUNTRY CO2 EMISSIONS PER YEAR GRAPH
    #------------------------------------------------------

    #text title
    st.subheader("CO2 Emissions per Country:")
    #------------------------------------------------


    # filter to CO2-by-country from the USA dataset
    co2_countries = df_individual[df_individual['Indicator'] == 'CO2 Emissions (Metric Tons)'].copy()

    # sort so each country's line is drawn in time order
    co2_countries = co2_countries.sort_values(['Country', 'Year'])

    # plot one line per country
    fig, ax = plt.subplots()  
    for country, dfc in co2_countries.groupby('Country'):
        ax.plot(dfc['Year'], dfc['Value'], alpha=0.4, color='black')

    # overlay the U.S. line on the same axes
    us = co2_countries[co2_countries['Country'] == 'United States'].sort_values('Year')
    ax.plot(us['Year'], us['Value'], color='blue', linewidth=2, label='United States', zorder=5)

    # legend on the right
    ax.legend(title='Country', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    # title and axis labels
    ax.set_title("Country CO$_2$ Emissions per Year (1751-2014)")
    ax.set_ylabel("Emissions (Metric Tonnes)")
    ax.set_xlabel("Year")

    # apply theme and caption
    my_theme(ax, fig, "Limited to reporting countries")

    # display in streamlit
    st.pyplot(fig)

    #--------------------------------------------------------------------
    # top 10 emission-producing countries in 2014 graph
    #--------------------------------------------------------------------

    #text title
    st.subheader("Top 10 Emitting Countries:")
    #--------------------------------------------------

    top_2014 = (
        df_individual[
            (df_individual['Indicator'] == 'CO2 Emissions (Metric Tons)') &
            (df_individual['Year'] == 2014)
        ].copy()
    )

    # rank by largest first
    top_2014['rank'] = top_2014['Value'].rank(method='dense', ascending=False).astype(int)

    # keep top 10 and order by rank
    top_10_count = top_2014[top_2014['rank'] <= 10].sort_values('rank')

    # countries to keep
    top_countries = top_10_count['Country'].tolist()

    # filter the long data to those countries / emissions / years >= 1900
    top10b_df = df_individual[
        (df_individual['Indicator'] == 'CO2 Emissions (Metric Tons)') &
        (df_individual['Country'].isin(top_countries)) &
        (df_individual['Year'] >= 1900)
    ].copy()

    # make a discrete viridis palette with exactly len(top_countries) colors
    from matplotlib.cm import get_cmap
    cmap = get_cmap('viridis', len(top_countries))
    color_map = {country: cmap(i) for i, country in enumerate(top_countries)}

    # plot
    fig, ax = plt.subplots()
    for country in top_countries:  
        dfc = top10b_df[top10b_df['Country'] == country].sort_values('Year')
        ax.plot(dfc['Year'], dfc['Value'], label=country, color=color_map[country])

    # force legend to only show top_10_count countries in ranked order
    handles, labels = ax.get_legend_handles_labels()
    ordered_labels = top_countries
    ordered_handles = [handles[labels.index(lbl)] for lbl in ordered_labels if lbl in labels]
    ax.legend(ordered_handles, ordered_labels, title='Country',
            loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    # title + subtitle
    ax.set_title("Top 10 Emissions-producing Countries in 2014 (1900–2014)\n"
                "Ordered by Emissions Produced in 2014", fontsize=16)

    # axes labels
    ax.set_ylabel("Emissions (Metric Tonnes)")
    ax.set_xlabel("Year")

    # apply theme + caption
    my_theme(ax, fig)

    # display in streamlit
    st.pyplot(fig)


    #-----------------------------------------------
    #Top 10 tile graph
    #------------------------------------------------

    #text title
    st.subheader("Top 10 Tile:")

    # pivot to Country rows, Year columns
    mat = top10b_df.pivot(index='Country', columns='Year', values='Value')

    # order countries by their 2014 value
    order = mat[2014].sort_values(ascending=False).index.tolist()
    mat = mat.loc[order]

    # reindex to cover full year range
    years = np.arange(1900, 2015)
    mat = mat.reindex(columns=years)

    # log transform and mask invalids
    log_mat = np.ma.masked_invalid(np.log(mat.to_numpy()))

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(log_mat, aspect='auto', cmap='viridis',
                extent=[years.min()-0.5, years.max()+0.5, -0.5, len(order)-0.5],
                origin='upper')

    # ticks and labels
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_xticks(np.arange(1900, 2015, 5))
    ax.set_xticklabels(np.arange(1900, 2015, 5), rotation=90)

    # titles
    ax.set_title("Top 10 CO$_2$ Emission-producing Countries", fontsize=16)
    ax.set_xlabel("Year")
    ax.set_ylabel("")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.10, pad=0.10)
    cbar.set_label("Ln(CO$_2$ Emissions (Metric Tonnes))", fontsize=12)

    plt.tight_layout()

    # display in Streamlit
    st.pyplot(fig)

    #-----------------------------------------
    #Facets graph
    #-----------------------------------------

    st.subheader("Facets:")

    # copy and normalize country names so US data aggregates consistently
    df = df_individual.copy()
    df['Country'] = df['Country'].replace({
        'United States of America': 'United States',
        'USA': 'United States',
        'US': 'United States'
    })

    # region flag for facets
    df['Region'] = np.where(df['Country'] == 'United States', 'United States', 'Rest of the World')

    # short indicator labels restricted to three indicators
    indicator_map = {
        "CO2 Emissions (Metric Tons)": "Emissions",
        "Energy Use (kg, oil-eq./capita)": "Energy",
        "GDP Growth/Capita (%)": "GDP",
        "Emissions": "Emissions",
        "Energy": "Energy",
        "GDP": "GDP",
    }
    df["Indicator_short"] = df["Indicator"].map(indicator_map)
    df = df[df["Indicator_short"].isin(["Emissions", "Energy", "GDP"])].copy()

    # facet orders
    row_order = ["Emissions", "Energy", "GDP"]
    col_order = ["Rest of the World", "United States"]

    # facet grid with shared x and per-row shared y
    g = sns.FacetGrid(
        df, row="Indicator_short", col="Region",
        row_order=row_order, col_order=col_order,
        sharex=True, sharey='row',
        height=3, aspect=1.5
    )

    # draw one thin line per country in each facet
    def draw_lines(data, **kwargs):
        for _, grp in data.sort_values("Year").groupby("Country"):
            plt.plot(grp["Year"], grp["Value"], color="black", linewidth=1)

    g.map_dataframe(draw_lines)

    # enforce same y-limits across both columns within each row
    for i, ind in enumerate(row_order):
        y = df.loc[df["Indicator_short"] == ind, "Value"].dropna()
        if not y.empty:
            ymin, ymax = y.min(), y.max()
            pad = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) + 1))
            for ax in g.axes[i, :]:
                ax.set_ylim(ymin - pad, ymax + pad)

    # remove default facet titles and axis labels
    g.set_titles("")
    g.set_axis_labels("", "")

    # shared x and y labels
    g.figure.text(0.5, 0.04, "Year", ha="center", va="center", fontsize=12)
    g.figure.text(0.02, 0.5, "Indicator Value", ha="center", va="center", rotation=90, fontsize=12)

    # overall title
    g.figure.suptitle("Distribution of Indicators by Year and Value", fontsize=16, y=0.98)

    # column headers centered above each column
    for i, col_name in enumerate(g.col_names):
        ax = g.axes[0, i]
        bb = ax.get_position()
        g.figure.text((bb.x0 + bb.x1) / 2, 0.90, col_name, ha="center", va="bottom", fontsize=14, weight="bold")

    # row labels on the right side
    for i, row_name in enumerate(g.row_names):
        ax = g.axes[i, -1]
        ax.text(1.02, 0.5, row_name, transform=ax.transAxes, ha="left", va="center", fontsize=12, rotation=-90)

    # grid and ticks styling
    for ax in g.axes.flat:
        ax.grid(True, color="black", linewidth=0.5, alpha=0.3)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0.06, 0.08, 0.98, 0.93])

    # render in streamlit
    st.pyplot(g.figure)


    #----------------------------------------
    #scatter plots
    #----------------------------------------

    st.subheader("CO2 Emissions and Temperature Side By Side Scatter Plots:")

    # filter to USA, years, and the two indicators
    df = df_individual.copy()

    # map full indicator names to short labels
    indicator_map = {
        "CO2 Emissions (Metric Tons)": "Emissions",
        "Emissions": "Emissions",
        "Temperature": "Temperature",
    }
    df["Indicator_short"] = df["Indicator"].map(indicator_map)

    # keep USA, 1980–2014, and only Emissions/Temperature
    us = df[
        (df["Country"] == "United States") &
        (df["Year"] >= 1980) & (df["Year"] <= 2014) &
        (df["Indicator_short"].isin(["Emissions", "Temperature"]))
    ].copy()

    # split into two panels
    panels = [us[us["Indicator_short"] == ind].copy() for ind in ["Emissions", "Temperature"]]

    # plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 6), sharex=True)

    for ax, d in zip(axes, panels):
        # white background
        ax.set_facecolor("white")

        # scatter points
        ax.scatter(d["Year"], d["Value"], s=16)

        # LOWESS smooth if enough points
        d_sorted = d.sort_values("Year")
        if len(d_sorted) >= 5:
            sm = lowess(d_sorted["Value"], d_sorted["Year"], frac=0.4, it=0, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], linewidth=2)

        # panel title from label, fallback to short indicator
        label_txt = d["Label"].iloc[0] if not d["Label"].isna().all() else d["Indicator_short"].iloc[0]
        ax.set_title(label_txt, fontsize=14)

    # x ticks
    xticks = np.arange(1980, 2015, 5)
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([str(x) for x in xticks], rotation=90, fontsize=12)

    # axis labels and tick sizes
    axes[-1].set_xlabel("Year", fontsize=12)
    for ax in axes:
        ax.tick_params(axis='y', labelsize=12)

    # figure title
    fig.suptitle("US Emissions and Temperatures (1980–2014)", fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # display in Streamlit
    st.pyplot(fig)

    #----------------------------------------
    #----------------------------------------
    #Stat Header
    st.header("Analysis")

    #-----------------------
    #mean and SD Display:
    #------------------------

    st.subheader("Mean & SD")


    #year range interactively
    year_min = int(df_individual["Year"].min())
    year_max = int(df_individual["Year"].max())
    start, end = st.slider(
        "Year range",
        min_value=year_min, max_value=year_max,
        value=(1980, 2014), step=1,
        help="Choose the years to include in the summary stats"
    )

    # compute stats for USA within the selected range
    subset = df_individual[
        (df_individual["Country"] == "United States") &
        (df_individual["Year"].between(start, end)) &
        (df_individual["Indicator"].isin(["CO2 Emissions (Metric Tons)", "Temperature"]))
    ].copy()

    # map names for display
    name_map = {
        "CO2 Emissions (Metric Tons)": "Emissions",
        "Temperature": "Temperature"
    }
    subset["Metric"] = subset["Indicator"].map(name_map)

    stats = (subset
            .groupby("Metric")["Value"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "Mean", "std": "SD"}))

    # pull values for the metric cards
    em_mean = stats.loc["Emissions", "Mean"] if "Emissions" in stats.index else float("nan")
    em_sd   = stats.loc["Emissions", "SD"]   if "Emissions" in stats.index else float("nan")
    t_mean  = stats.loc["Temperature", "Mean"] if "Temperature" in stats.index else float("nan")
    t_sd    = stats.loc["Temperature", "SD"]   if "Temperature" in stats.index else float("nan")

    # metric cards
    c1, c2 = st.columns(2)
    c1.metric("Emissions — Mean (Metric Tonnes)", f"{em_mean:,.0f}", delta=f"SD {em_sd:,.0f}")
    c2.metric("Temperature — Mean (°C)", f"{t_mean:,.2f}", delta=f"SD {t_sd:,.2f}")

    # a tidy table + download
    st.markdown("**Details**")
    stats_display = stats.copy()
    stats_display["Mean"] = stats_display["Mean"].round(3)
    stats_display["SD"]   = stats_display["SD"].round(3)
    st.table(stats_display)

    #-----------------------------------
    #Correlation Coefficent Display
    #-----------------------------------

    # build the wide_US DataFrame

    us_data = df_individual[
        (df_individual['Country'] == 'United States') &
        (df_individual['Indicator'].isin(['CO2 Emissions (Metric Tons)', 'Temperature']))
    ].copy()

    #pivot to wide format: Year as index
    wide_US = us_data.pivot(index='Year', columns='Indicator', values='Value')

    #rename columns for easier access
    wide_US = wide_US.rename(columns={
        'CO2 Emissions (Metric Tons)': 'Emissions',
        'Temperature': 'Temperature'
    })

    #sort by year
    wide_US = wide_US.sort_index()

    #Correlation coefficent display
    r = wide_US["Emissions"].corr(wide_US["Temperature"], method="pearson")

    st.subheader("Emissions vs. Temperature Correlation")
    st.metric(
        label="Pearson Correlation Coefficient (Emissions vs. Temperature)",
        value=f"r = {r:.3f}",
        delta=(
            "Very Strongly Associated" if abs(r) >= 0.8 else
            "Strongly Associated" if abs(r) >= 0.6 else
            "Clearly Associated" if abs(r) >= 0.4 else
            "Weakly Associated" if abs(r) >= 0.2 else
            "Very Weakly Associated"
        )
    )

    #---------------------------------------
    #correlation Graph
    #-----------------------------------

    #subheader
    st.subheader("See the Correlation:")

    # restrict to analysis window
    wide_us_period = wide_US.loc[1980:2014].copy()

    # drop rows with missing values
    cleaned_wide_US = wide_us_period.dropna(subset=["Emissions", "Temperature"])

    # z-score the two series
    x = (cleaned_wide_US["Emissions"] - cleaned_wide_US["Emissions"].mean()) / cleaned_wide_US["Emissions"].std(ddof=1)
    y = (cleaned_wide_US["Temperature"] - cleaned_wide_US["Temperature"].mean()) / cleaned_wide_US["Temperature"].std(ddof=1)

    # fit simple linear regression on scaled variables
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    # plot scatter with regression line
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, s=30, alpha=0.9)
    ax.plot(x_line, y_line, linewidth=2)

    # titles and axis labels
    ax.set_title("US CO$_2$ Emissions and Temperature (1980–2014)", fontsize=16)
    ax.set_xlabel("Scaled Emissions (Metric Tonnes)", fontsize=14)
    ax.set_ylabel("Scaled Temperature (Fahrenheit)", fontsize=14)

    # styling
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="black", linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # render in Streamlit
    plt.tight_layout()
    st.pyplot(fig)

    #----------------------------
    #----------------------------
    #Summary Plot Header
    st.header("Summary Plot")

    #Summary Plot

    # world CO2 line (1751–2014)
    def fig_world_emissions(df_long):
        world = (
            df_long[df_long['Indicator'] == 'CO2 Emissions (Metric Tons)']
            .groupby('Year', as_index=False)['Value'].sum()
            .rename(columns={'Value': 'Emissions'})
            .sort_values('Year')
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(world['Year'], world['Emissions'], linewidth=1.5)
        ax.set_title("World CO$_2$ Emissions per Year (1751-2014)")
        ax.set_ylabel("Emissions (Metric Tonnes)")
        ax.set_xlabel("Year")
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.grid(True, color='black', linewidth=0.5, alpha=0.3)
        return fig

    # top 10 emitters heatmap (1900–2014), ranked by 2014
    def fig_top10_heatmap(df_long):
        top_2014 = (
            df_long[
                (df_long['Indicator'] == 'CO2 Emissions (Metric Tons)') &
                (df_long['Year'] == 2014)
            ].copy()
        )
        top_2014['rank'] = top_2014['Value'].rank(method='dense', ascending=False).astype(int)
        top_countries = top_2014[top_2014['rank'] <= 10].sort_values('rank')['Country'].tolist()

        top10b_df = df_long[
            (df_long['Indicator'] == 'CO2 Emissions (Metric Tons)') &
            (df_long['Country'].isin(top_countries)) &
            (df_long['Year'] >= 1900)
        ].copy()

        mat = top10b_df.pivot(index='Country', columns='Year', values='Value')
        order = mat[2014].sort_values(ascending=False).index.tolist()
        mat = mat.loc[order]
        years = np.arange(1900, 2015)
        mat = mat.reindex(columns=years)

        log_mat = np.ma.masked_invalid(np.log(mat.to_numpy()))

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(
            log_mat, aspect='auto', cmap='viridis',
            extent=[years.min()-0.5, years.max()+0.5, -0.5, len(order)-0.5],
            origin='upper'
        )
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
        ax.set_xticks(np.arange(1900, 2015, 5))
        ax.set_xticklabels(np.arange(1900, 2015, 5), rotation=90)
        ax.set_title("Top 10 CO$_2$ Emission-producing Countries", fontsize=16)
        ax.set_xlabel("Year")
        ax.set_ylabel("")
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.10, pad=0.10)
        cbar.set_label("Ln(CO$_2$ Emissions (Metric Tonnes))", fontsize=12)
        plt.tight_layout()
        return fig

    # US facet (three indicators by Region) rendered onto a single figure
    def fig_us_facet(df_long):
        df = df_long.copy()
        df['Country'] = df['Country'].replace({
            'United States of America': 'United States',
            'USA': 'United States',
            'US': 'United States'
        })
        df['Region'] = np.where(df['Country'] == 'United States', 'United States', 'Rest of the World')
        indicator_map = {
            "CO2 Emissions (Metric Tons)": "Emissions",
            "Energy Use (kg, oil-eq./capita)": "Energy",
            "GDP Growth/Capita (%)": "GDP",
            "Emissions": "Emissions",
            "Energy": "Energy",
            "GDP": "GDP",
        }
        df["Indicator_short"] = df["Indicator"].map(indicator_map)
        df = df[df["Indicator_short"].isin(["Emissions", "Energy", "GDP"])].copy()

        row_order = ["Emissions", "Energy", "GDP"]
        col_order = ["Rest of the World", "United States"]

        g = sns.FacetGrid(
            df, row="Indicator_short", col="Region",
            row_order=row_order, col_order=col_order,
            sharex=True, sharey='row', height=3.3, aspect=1.5
        )
        def draw_lines(data, **kwargs):
            for _, grp in data.sort_values("Year").groupby("Country"):
                plt.plot(grp["Year"], grp["Value"], color="black", linewidth=1)

        g.map_dataframe(draw_lines)

        for i, ind in enumerate(row_order):
            y = df.loc[df["Indicator_short"] == ind, "Value"].dropna()
            if not y.empty:
                ymin, ymax = y.min(), y.max()
                pad = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) + 1))
                for ax in g.axes[i, :]:
                    ax.set_ylim(ymin - pad, ymax + pad)

        g.set_titles("")
        g.set_axis_labels("", "")
        g.figure.text(0.5, 0.04, "Year", ha="center", va="center", fontsize=12)
        g.figure.text(0.02, 0.5, "Indicator Value", ha="center", va="center", rotation=90, fontsize=12)
        g.figure.suptitle("Distribution of Indicators by Year and Value", fontsize=16, y=0.98)

        for i, col_name in enumerate(g.col_names):
            ax = g.axes[0, i]
            bb = ax.get_position()
            g.figure.text((bb.x0 + bb.x1) / 2, 0.90, col_name, ha="center", va="bottom", fontsize=14, weight="bold")

        for ax in g.axes.flat:
            ax.grid(True, color="black", linewidth=0.5, alpha=0.3)
            ax.set_facecolor("white")
            ax.tick_params(labelsize=10)

        plt.tight_layout(rect=[0.06, 0.08, 0.98, 0.93])
        return g.figure

    # US scaled scatter with regression line (1980–2014)
    def fig_us_scaled_scatter(wide_us):
        wide_us_period = wide_us.loc[1980:2014].copy()
        cleaned = wide_us_period.dropna(subset=["Emissions", "Temperature"])
        x = (cleaned["Emissions"] - cleaned["Emissions"].mean()) / cleaned["Emissions"].std(ddof=1)
        y = (cleaned["Temperature"] - cleaned["Temperature"].mean()) / cleaned["Temperature"].std(ddof=1)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x, y, s=30, alpha=0.9)
        ax.plot(x_line, y_line, linewidth=2)
        ax.set_title("US CO$_2$ Emissions and Temperature (1980–2014)", fontsize=16)
        ax.set_xlabel("Scaled Emissions (Metric Tonnes)", fontsize=14)
        ax.set_ylabel("Scaled Temperature (Fahrenheit)", fontsize=14)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.grid(True, color="black", linewidth=0.5, alpha=0.3)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        plt.tight_layout()
        return fig

    # build wide_US from df_individual for the scatter plot
    us_data = (
        df_individual[
            (df_individual["Country"] == "United States") &
            (df_individual["Indicator"].isin(["CO2 Emissions (Metric Tons)", "Temperature"]))
        ].copy()
    )
    wide_US = (
        us_data
        .pivot(index="Year", columns="Indicator", values="Value")
        .rename(columns={"CO2 Emissions (Metric Tons)": "Emissions", "Temperature": "Temperature"})
        .sort_index()
    )

    # render 2x2 grid with Streamlit columns
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_world_emissions(df_individual))
    with c2:
        st.pyplot(fig_top10_heatmap(df_individual))
    with c1:
        st.pyplot(fig_us_facet(df_individual))
    with c2:
        st.pyplot(fig_us_scaled_scatter(wide_US))

with tab2:
    st.markdown("Jump to: [Visuals](#visuals) | [Analysis](#analysis) | [Summary Plot](#summary-plot)")
    st.header("Intro")
    st.write(
        """This dashboard presents the results of a group case study focused on Norway’s role in global carbon dioxide (CO₂) emissions and the potential connections between emissions, temperature changes, and natural disasters.
        Norway is a particularly interesting case because, while it is a relatively small country, it has historically been a major producer and exporter of fossil fuels. 
        Understanding Norway’s own emissions profile, alongside broader global trends, helps us evaluate how even smaller nations contribute to climate change in the context of global energy markets.  \n\n"""

        "**The study addresses two main research questions:**  \n"
        "**1)** How have Norway’s CO₂ emissions changed over time, and how does Norway compare to the rest of the world?  \n"
        "**2)** Are CO₂ emissions, temperature, and natural disasters within Norway associated?"
    )
    st.header("Visuals")

    st.subheader("CO2 Emissions Per Country:")
    #-------------------------------
    #CO2 Emissions Graph
    #-------------------------------
    # filter to CO2 emissions
    co2_data = df_group[df_group['Indicator'] == 'CO2 Emissions (Metric Tons)'].copy()

    # convert to billions of metric tons
    co2_data['Value_billion'] = co2_data['Value'] / 1e9

    # figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # plot all countries with low transparency
    sns.lineplot(
        data=co2_data.sort_values(['Country', 'Year']),
        x='Year', y='Value_billion',
        units='Country', estimator=None,
        color='gray', linewidth=1, alpha=0.3,
        legend=False, ax=ax
    )

    # highlight Norway
    norway_data = co2_data[co2_data['Country'] == 'Norway'].sort_values('Year')
    sns.lineplot(
        data=norway_data,
        x='Year', y='Value_billion',
        color='red', linewidth=2.5, label='Norway', ax=ax
    )

    # label at the last Norway point
    if not norway_data.empty:
        last_year = float(norway_data['Year'].max())
        last_val = float(norway_data.loc[norway_data['Year'] == last_year, 'Value_billion'].iloc[0])
        ax.text(last_year + 0.5, last_val, 'Norway', fontsize=10, va='center', ha='left')
        ax.set_xlim(co2_data['Year'].min(), last_year + 5)

    # titles and styling
    ax.set_title('CO₂ Emissions Over Time (All Countries with Norway Highlighted)', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('CO₂ Emissions (Billions of Metric Tons)', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Top 10 Emistting Countries:")
    #-------------------------------
    #Top 10 Graph
    #-------------------------------

    # use all available years and convert to billions
    emissions_all_years = co2_data.copy()
    emissions_all_years['Value_billion'] = emissions_all_years['Value'] / 1e9

    # identify the top 10 by maximum observed emissions
    latest_year_emissions = (
        emissions_all_years.dropna(subset=['Value_billion'])
        .groupby('Country')['Value_billion']
        .max()
    )
    top_10_countries = latest_year_emissions.nlargest(10).index.tolist()

    # figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # loop through each top country
    for country in top_10_countries:
        country_data = emissions_all_years[emissions_all_years['Country'] == country].sort_values('Year')
        if country_data.empty:
            continue
        ax.plot(country_data['Year'], country_data['Value_billion'],
                linewidth=2, label=country)

        # label at last data point
        last_year = country_data['Year'].max()
        last_val = float(country_data.loc[country_data['Year'] == last_year, 'Value_billion'].iloc[0])
        ax.text(last_year + 0.5, last_val, country,
                fontsize=9, va='center', ha='left')

    # titles and labels
    ax.set_title(
        f"CO₂ Emissions by Top 10 Countries ({int(co2_data['Year'].min())}–2024)",
        fontsize=16
    )
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("CO₂ Emissions (Billions of Metric Tons)", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(co2_data['Year'].min(), 2024)

    # styling
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Top 10 Tile:")
    #-------------------------------
    #Top 10 Tile
    #-------------------------------

    # filter to CO2 emissions
    co2_data = df_group[df_group['Indicator'] == 'CO2 Emissions (Metric Tons)'].copy()

    # ensure numeric year (if needed)
    co2_data['Year'] = pd.to_numeric(co2_data['Year'], errors='coerce')

    # top 10 countries by 2014 emissions
    co2_2014 = co2_data[co2_data['Year'] == 2014.0].copy()
    co2_top_10_2014 = co2_2014.nlargest(10, 'Value').copy()
    top_10_countries_2014 = co2_top_10_2014['Country'].tolist()

    # keep only those countries
    co2_top_10_data = co2_data[co2_data['Country'].isin(top_10_countries_2014)].copy()

    # pivot Year x Country, fill, log-transform
    pivot = co2_top_10_data.pivot(index='Year', columns='Country', values='Value')
    pivot_filled = pivot.fillna(0)
    pivot_log = np.log(pivot_filled + 1)

    # back to long for plotnine
    pivot_log_long = pivot_log.reset_index().melt(
        id_vars='Year', var_name='Country', value_name='log_emissions'
    )

    # order countries by 2014 emissions (highest first)
    country_order = co2_top_10_2014.sort_values('Value', ascending=False)['Country'].tolist()
    pivot_log_long['Country'] = pd.Categorical(pivot_log_long['Country'],
                                            categories=country_order, ordered=True)

    # compute x-axis breaks every 20 years
    year_min = int(pivot_log_long['Year'].min())
    year_max = int(pivot_log_long['Year'].max())
    year_breaks = np.arange(year_min, year_max + 1, 20)

    # build plot
    tile_plot = (
        ggplot(pivot_log_long, aes(x='Year', y='Country', fill='log_emissions'))
        + geom_tile()
        + scale_fill_gradient(low='#f7fbff', high='#08306b')
        + labs(
            title='CO₂ Emissions Over Time for Top 10 Countries (Ordered by 2014 Emissions)',
            x='Year', y='Country', fill='Log(Emissions + 1)'
        )
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            figure_size=(12, 8)
        )
        + scale_x_continuous(breaks=year_breaks)
        + scale_y_discrete(limits=country_order[::-1])  # highest at top
    )

    # render in Streamlit
    fig = tile_plot.draw()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Facets:")
    #-------------------------------
    #Facets
    #-------------------------------

    # keep only Emissions / Energy / GDP for Norway visuals
    keep_indicators = [
        "CO2 Emissions (Metric Tons)",
        "Energy Use (kg, oil-eq./capita)",
        "GDP Growth/Capita (%)"
    ]
    data_long_filtered = df_group[df_group["Indicator"].isin(keep_indicators)].copy()

    # rename indicators to short labels
    indicator_mapping = {
        "CO2 Emissions (Metric Tons)": "Emissions",
        "Energy Use (kg, oil-eq./capita)": "Energy",
        "GDP Growth/Capita (%)": "GDP"
    }
    data_long_filtered["Indicator"] = data_long_filtered["Indicator"].map(indicator_mapping)

    # ensure numeric and drop missing
    for col in data_long_filtered.columns:
        if col not in ["Country", "Label", "Indicator", "Region"]:
            data_long_filtered[col] = pd.to_numeric(data_long_filtered[col], errors="coerce")
    data_long_filtered = data_long_filtered.dropna().reset_index(drop=True)

    # facet wrap (Norway highlighted)
    facet_wrap_plot = (
        ggplot(data_long_filtered, aes(x="Year", y="Value", group="Country"))
        + geom_line(alpha=0.2)
        + geom_line(
            data=data_long_filtered.query('Country == "Norway"'),
            mapping=aes(x="Year", y="Value", color="Country")
        )
        + scale_colour_manual(values=["red"])
        + labs(
            title="Distribution of Indicators by Year and Value (Norway Highlighted)",
            y="Indicator Value"
        )
        + theme(strip_text=element_text(size=12))
        + facet_wrap("~Indicator", scales="free_y", ncol=1)
        + coord_cartesian(xlim=(1750, None))
    )

    # facet grid (Norway vs Rest of the World)
    facet_grid_plot = (
        ggplot(data_long_filtered, aes(x="Year", y="Value", group="Country"))
        + geom_line(alpha=0.2)
        + geom_line(
            data=data_long_filtered.query('Region == "Norway"'),
            mapping=aes(x="Year", y="Value", color="Region")
        )
        + scale_colour_manual(values=["red"])
        + facet_grid("Indicator ~ Region", scales="free_y")
        + labs(
            title="Distribution of Indicators by Year and Value (Norway vs Rest of the World)",
            y="Indicator Value"
        )
        + theme(strip_text=element_text(size=12))
        + coord_cartesian(xlim=(1750, None))
    )

    # render 
    fig1 = facet_wrap_plot.draw()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    fig2 = facet_grid_plot.draw()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


    st.subheader("CO2 and Temperature Side By Side")
    #-------------------------------
    #Side by Side Scatters
    #-------------------------------
    # Filter to only Norway
    norway_data_filtered = df_group[df_group['Country'] == 'Norway'].copy()

    # Keep only Emissions and Temperature
    indicators_to_keep = ['CO2 Emissions (Metric Tons)', 'Temperature']
    norway_indicators_filtered = norway_data_filtered[
        norway_data_filtered['Indicator'].isin(indicators_to_keep)
    ].copy()

    # Restrict to years 1980–2014
    norway_emissions_temp_filtered = norway_indicators_filtered[
        (norway_indicators_filtered['Year'] >= 1980) &
        (norway_indicators_filtered['Year'] <= 2014)
    ].copy()
    #plot
    fig = (
        ggplot(norway_emissions_temp_filtered, aes(x='Year', y='Value', color='Indicator'))
        + geom_point()
        + geom_smooth(method='loess', se=False)
        + facet_wrap('~Indicator', scales='free_y', ncol=1)
        + labs(title="CO₂ Emissions and Temperature in Norway (1980–2014)", x="Year", y="Value")
    ).draw()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.subheader("Emissions vs. Temperature")
    #-------------------------------
    #Emissions vs Temp Scatter
    #-------------------------------
    # Filter for Norway
    norway_data = df_group[df_group['Country'] == 'Norway'].copy()

    # Further filter for Emissions and Temperature indicators
    indicators_to_keep = ['CO2 Emissions (Metric Tons)', 'Temperature']
    norway_indicators = norway_data[norway_data['Indicator'].isin(indicators_to_keep)].copy()

    # Filter for years between 1980 and 2014
    norway_emissions_temp = norway_indicators[
        (norway_indicators['Year'] >= 1980) & (norway_indicators['Year'] <= 2014)
    ].copy()

    # Reshape the DataFrame from long to wide format
    norway_pivot = norway_emissions_temp.pivot(index='Year', columns='Indicator', values='Value')

    # Final scatter plot with regression line and labels
    final_scatter_plot = (
        ggplot(norway_pivot, aes(x='CO2 Emissions (Metric Tons)', y='Temperature'))
        + geom_point()
        + geom_smooth(method='lm', se=False)
        + labs(
            title="CO2 Emissions vs. Temperature in Norway (1980-2014)",
            x="CO2 Emissions (Metric Tons)",
            y="Temperature (°C)"
        )
    )

    fig = final_scatter_plot.draw()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    #-----------------------------
    #----------------------------
    #Analysis
    st.header("Analysis")

    #-----------------------------
    #Mean and SD
    #-------------------------
    st.subheader("Mean & SD")

    # year slider based on available Norway years in the group dataset
    nor_year_min = int(df_group.loc[df_group["Country"] == "Norway", "Year"].min())
    nor_year_max = int(df_group.loc[df_group["Country"] == "Norway", "Year"].max())
    start_nor, end_nor = st.slider(
        "Year range (Norway)",
        min_value=nor_year_min, max_value=nor_year_max,
        value=(1980, 2014), step=1,
        help="Choose years to include in the Norway summary stats"
    )

    # filter to Norway + indicators within the chosen range
    subset_nor = df_group[
        (df_group["Country"] == "Norway") &
        (df_group["Year"].between(start_nor, end_nor)) &
        (df_group["Indicator"].isin(["CO2 Emissions (Metric Tons)", "Temperature"]))
    ].copy()

    # map to simple metric names
    name_map_nor = {
        "CO2 Emissions (Metric Tons)": "Emissions",
        "Temperature": "Temperature"
    }
    subset_nor["Metric"] = subset_nor["Indicator"].map(name_map_nor)

    # compute mean/sd
    stats_nor = (subset_nor
                .groupby("Metric")["Value"]
                .agg(["mean", "std"])
                .rename(columns={"mean": "Mean", "std": "SD"}))

    # extract values for metric cards
    em_mean_n = stats_nor.loc["Emissions", "Mean"] if "Emissions" in stats_nor.index else float("nan")
    em_sd_n   = stats_nor.loc["Emissions", "SD"]   if "Emissions" in stats_nor.index else float("nan")
    t_mean_n  = stats_nor.loc["Temperature", "Mean"] if "Temperature" in stats_nor.index else float("nan")
    t_sd_n    = stats_nor.loc["Temperature", "SD"]   if "Temperature" in stats_nor.index else float("nan")

    # metric cards
    c1, c2 = st.columns(2)
    c1.metric("Emissions — Mean (Metric Tonnes)", f"{em_mean_n:,.0f}", delta=f"SD {em_sd_n:,.0f}")
    c2.metric("Temperature — Mean (°C)", f"{t_mean_n:,.2f}", delta=f"SD {t_sd_n:,.2f}")

    # table
    st.markdown("**Details**")
    stats_display_nor = stats_nor.copy()
    stats_display_nor["Mean"] = stats_display_nor["Mean"].round(3)
    stats_display_nor["SD"]   = stats_display_nor["SD"].round(3)
    st.table(stats_display_nor)

    # correlation coefficent display
    st.subheader("Emissions vs. Temperature Correlation — Norway")

    wide_nor = subset_nor.pivot(index="Year", columns="Indicator", values="Value")

    rename_map = {}
    for col in wide_nor.columns:
        if col.strip().lower() in ("co2 emissions (metric tons)", "co₂ emissions (metric tons)"):
            rename_map[col] = "Emissions"
    for col in wide_nor.columns:
        if "temperature" in col.strip().lower():
            rename_map[col] = "Temperature"

    wide_nor = wide_nor.rename(columns=rename_map).sort_index()

    needed = {"Emissions", "Temperature"}
    if not needed.issubset(set(wide_nor.columns)):
        st.warning(f"Missing expected columns for correlation. Found: {list(wide_nor.columns)}")
    else:
        wide_nor = wide_nor.dropna(subset=["Emissions", "Temperature"], how="any")
        if len(wide_nor) < 2:
            st.warning("Not enough Norway rows to compute correlation in the selected range.")
        else:
            r_nor = wide_nor["Emissions"].corr(wide_nor["Temperature"], method="pearson")
            st.metric(
                label="Pearson Correlation Coefficient (Emissions vs. Temperature)",
                value=f"r = {r_nor:.3f}",
                delta=(
                    "Very Strongly Associated" if abs(r_nor) >= 0.8 else
                    "Strongly Associated"      if abs(r_nor) >= 0.6 else
                    "Clearly Associated"       if abs(r_nor) >= 0.4 else
                    "Weakly Associated"        if abs(r_nor) >= 0.2 else
                    "Very Weakly Associated"
                )
            )



    st.header("Summary Plot")
        # --- Summary Plot (Norway) ---

    def fig_summary_norway(df_long):
        # ensure numeric year
        df = df_long.copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        # panel A: CO2 by country (Norway highlighted)
        co2_data = df[df["Indicator"] == "CO2 Emissions (Metric Tons)"].dropna(subset=["Year", "Value"]).copy()
        latest_year = int(co2_data["Year"].max())

        # 2x2 grid (bottom-left is a sub-grid)
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        sub_gs = gs[1, 0].subgridspec(2, 1, hspace=0.25)
        ax10 = fig.add_subplot(sub_gs[0, 0])  # emissions
        ax20 = fig.add_subplot(sub_gs[1, 0])  # temperature
        ax11 = fig.add_subplot(gs[1, 1])      # scaled scatter

        # panel A
        for _, grp in co2_data.groupby("Country"):
            ax00.plot(grp["Year"], grp["Value"], color="gray", alpha=0.3, linewidth=1)
        nor_co2 = co2_data[co2_data["Country"] == "Norway"].sort_values("Year")
        if not nor_co2.empty:
            ax00.plot(nor_co2["Year"], nor_co2["Value"], color="red", linewidth=2.5, label="Norway")
            ax00.legend(frameon=False, loc="upper left")
        ax00.set_title("CO₂ Emissions by Country (1751–2014)")
        ax00.set_xlabel("Year")
        ax00.set_ylabel("CO₂ Emissions (Metric Tonnes)")
        ax00.grid(True, linestyle="--", alpha=0.5)

        # panel B (heatmap top10 by latest year)
        top10_countries = (
            co2_data[co2_data["Year"] == latest_year]
            .sort_values("Value", ascending=False)
            .head(10)["Country"]
            .tolist()
        )
        top10_df = co2_data[co2_data["Country"].isin(top10_countries)].copy()
        pivot_top10 = top10_df.pivot(index="Country", columns="Year", values="Value")
        pivot_top10 = pivot_top10.loc[top10_countries]
        sns.heatmap(np.log1p(pivot_top10), cmap="Blues", ax=ax01, cbar_kws={"label": "log(Emissions + 1)"})
        ax01.set_title(f"Top 10 CO₂ Emitters ({latest_year}) and History")
        ax01.set_xlabel("Year")
        ax01.set_ylabel("Country")

        # Norway 1980–2014 subset
        nor = df[df["Country"] == "Norway"].copy()
        nor_1980_2014 = nor[(nor["Year"] >= 1980) & (nor["Year"] <= 2014)].copy()
        nor_co2_y = nor_1980_2014[nor_1980_2014["Indicator"] == "CO2 Emissions (Metric Tons)"]
        nor_tmp_y = nor_1980_2014[nor_1980_2014["Indicator"].str.contains("Temperature", case=False, na=False)]

        # panel C top: emissions with LOWESS
        if not nor_co2_y.empty:
            sns.regplot(
                x="Year", y="Value", data=nor_co2_y, ax=ax10,
                scatter=True, lowess=True, color="firebrick",
                scatter_kws={"s": 35, "alpha": 0.8}, line_kws={"linewidth": 2}
            )
        ax10.set_title("CO₂ Emissions (Norway, 1980–2014)")
        ax10.set_xlabel("")
        ax10.set_ylabel("")
        ax10.grid(True, linestyle="--", alpha=0.4)

        # panel C bottom: temperature with LOWESS
        if not nor_tmp_y.empty:
            sns.regplot(
                x="Year", y="Value", data=nor_tmp_y, ax=ax20,
                scatter=True, lowess=True, color="deepskyblue",
                scatter_kws={"s": 35, "alpha": 0.8}, line_kws={"linewidth": 2}
            )
        ax20.set_title("Temperature (Norway, 1980–2014)")
        ax20.set_xlabel("Year")
        ax20.set_ylabel("")
        ax20.grid(True, linestyle="--", alpha=0.4)

        # panel D: scaled scatter
        nor_wide = nor_1980_2014.pivot(index="Year", columns="Indicator", values="Value")
        col_map = {}
        for c in nor_wide.columns:
            cl = c.strip().lower()
            if "co2 emissions" in cl or "co₂ emissions" in cl:
                col_map[c] = "Emissions"
            if "temperature" in cl:
                col_map[c] = "Temperature"
        nor_wide = nor_wide.rename(columns=col_map)

        if {"Emissions", "Temperature"}.issubset(nor_wide.columns):
            em = nor_wide["Emissions"].astype(float)
            tp = nor_wide["Temperature"].astype(float)
            em_z = (em - em.mean()) / em.std(ddof=1)
            tp_z = (tp - tp.mean()) / tp.std(ddof=1)
            sns.regplot(x=em_z, y=tp_z, ax=ax11, color="black",
                        scatter_kws={"s": 35, "alpha": 0.8},
                        line_kws={"linewidth": 2})
            ax11.set_xlabel("Scaled CO₂ Emissions")
            ax11.set_ylabel("Scaled Temperature")
            ax11.set_title("Norway CO₂ Emissions vs. Temperature (1980–2014)")
            ax11.grid(True, linestyle="--", alpha=0.4)
        else:
            ax11.text(0.5, 0.5, "Missing Emissions/Temperature columns", ha="center", va="center")
            ax11.set_axis_off()

        fig.suptitle("Summary of Findings — Norway", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        return fig
    fig = fig_summary_norway(df_group)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

