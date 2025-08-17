import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns



# load the two dashboards’ datasets
df_individual = pd.read_csv("wrangled_data_individual.csv")  # USA
df_group = pd.read_csv("wrangled_data_group.csv")            # Norway

# ensure numeric year for grouping/plotting
for _df in (df_individual, df_group):
    if "Year" in _df.columns:
        _df["Year"] = pd.to_numeric(_df["Year"], errors="coerce")

#Create tabs at the top with linked markdowns to headers


tab1, tab2 = st.tabs(["Individual Project: USA", "Group Project: Norway"])

with tab1:
    st.markdown("Jump to: [Visuals](#Visuals) | [Statistics](#Statistics)")
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


#------------------------------------------------------
#WORLD CO2 EMISSION PER YEAR GRAPH
#------------------------------------------------------

#text title
st.write("**World CO2:**")
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
st.write("**CO2 Emissions per Country:**")
#------------------------------------------------


# filter to CO2-by-country from the USA dataset
co2_countries = df_individual[df_individual['Indicator'] == 'CO2 Emissions (Metric Tons)'].copy()

# sort so each country's line is drawn in time order
co2_countries = co2_countries.sort_values(['Country', 'Year'])

# plot one line per country (light gray/black background lines)
fig, ax = plt.subplots()  # you can pass figsize=(10,6) if you want a specific size
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
st.write("**Top 10 Emitting Countries:**")
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
st.write("**Top 10 tile**")

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
g.fig.text(0.5, 0.04, "Year", ha="center", va="center", fontsize=12)
g.fig.text(0.02, 0.5, "Indicator Value", ha="center", va="center", rotation=90, fontsize=12)

# overall title
g.fig.suptitle("Distribution of Indicators by Year and Value", fontsize=16, y=0.98)

# column headers centered above each column
for i, col_name in enumerate(g.col_names):
    ax = g.axes[0, i]
    bb = ax.get_position()
    g.fig.text((bb.x0 + bb.x1) / 2, 0.90, col_name, ha="center", va="bottom", fontsize=14, weight="bold")

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
st.pyplot(g.fig)

#----------------------------------------
#----------------------------------------
#Stat Header
st.header("Statistics")

with tab2:
    st.markdown("Jump to: [Visuals](#Visuals) | [Statistics](#Statistics)")
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
    st.header("Statistics")
