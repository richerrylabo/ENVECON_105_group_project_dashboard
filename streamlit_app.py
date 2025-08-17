import streamlit as st
import matplotlib.pyplot as plt


st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
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

# aggregate world CO2 emissions by year
world = (
    data_long[data_long['Indicator'] == 'CO2 Emissions (Metric Tons)']
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
