import streamlit as st
from streamlit.components.v1 import html
import requests
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import plotly.express as px
from babel.dates import format_date
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
import polars as pl
import time
import re
import gc
from PIL import Image


st.set_page_config(layout = "wide") 
logo = Image.open("warchat_logo.png")
st.logo(logo)

 
st.markdown(
    """
    <style>
    div[data-testid="stTabs"] button {
        font-size: 22px !important;  /* Adjust the font size */
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["📈 WarChart", "🤖 WarChat"])

font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 20px;
}
</style>
"""

st.write(font_css, unsafe_allow_html=True)


sidebar_placeholder = st.sidebar.empty()
with tab1:
    @st.cache_data
    def load_data():
        gdf = gpd.read_file("http://127.0.0.1:8000/api/events-json")
        gdf.drop(columns='geometry', inplace=True)
        gdf["fatalities"] = gdf["fatalities"].astype(int, errors='ignore')
        gdf["year"] = gdf["year"].astype(int, errors='ignore')
        gdf["latitude"] = gdf["latitude"].astype(float)
        gdf["longitude"] = gdf["longitude"].astype(float)
        
        return gdf


    df = load_data()
 
    # Sidebar filters
    with sidebar_placeholder:
        st.sidebar.header("Filter Options")
        event_type = st.sidebar.selectbox("Select Event Type", options=['Any'] + df['event_type'].unique().tolist())
        actor1 = st.sidebar.selectbox("Select Actor 1", options=['Any'] + df['actor1'].unique().tolist())
        actor2 = st.sidebar.selectbox("Select Actor 2", options=['Any'] + df['actor2'].unique().tolist())
        admin1 = st.sidebar.selectbox("Select Region", options=['Any'] + df['admin1'].unique().tolist())

        # Filters
        filtered_gdf = df.copy()
        if event_type != "Any":
            filtered_df = filtered_gdf[filtered_gdf['event_type'] == event_type]
        if actor1 != "Any":
            filtered_gdf = filtered_gdf[filtered_gdf['actor1'] == actor1]
        if actor2 != "Any":
            filtered_gdf = filtered_gdf[filtered_gdf['actor2'] == actor2]
        if admin1 != "Any":
            filtered_gdf = filtered_gdf[filtered_gdf['admin1'] == admin1]
            map_center = [filtered_gdf['latitude'].mean(), filtered_gdf['longitude'].mean()]


    update_datetime = datetime.now().strftime("%Y-%m-%d")


    # Stats display
    col1, col2, col3 = st.columns(spec=3, gap="small", vertical_alignment="bottom", border=False)
    total_events = len(filtered_gdf)
    total_fatalities = filtered_gdf['fatalities'].sum() 


    col1.markdown(
        f'<p style="font-size:18px; font-weight:bold;">Total Events: '
        f'<span style="color:red;">{total_events:,}</span></p>',
        unsafe_allow_html=True
    )
    col2.markdown(
        f'<p style="font-size:18px; font-weight:bold;">Civilian Fatalities: '
        f'<span style="color:red;">{total_fatalities:,}</span></p>',
        unsafe_allow_html=True
    )

    col3.markdown(
        f'<p style="font-size:18px; font-weight:bold;">Last Updated: '
        f'<span style="color:white;">{update_datetime}</span></p>',
        unsafe_allow_html=True
    )


    # Map and charts
    colmap, col_plot = st.columns([6, 4], gap="small", vertical_alignment="bottom", border=False)

    sdn_geojson_path ="data/sdn_adm0.geojson"
    with colmap:
        if filtered_gdf.latitude is not None:

            map_center = [filtered_gdf['latitude'].mean(), filtered_gdf['longitude'].mean()]

            if not np.isnan(map_center).any():
                m = folium.Map(location=map_center, zoom_start=4)

                folium.GeoJson(
                    sdn_geojson_path,
                    name="Sudan Border",
                    style_function=lambda feature: {
                        "fillColor": "transparent",  # No fill color
                        "color": "black",  # Border color
                        "weight": 2  # Border thickness
                    }
                ).add_to(m)

                marker_cluster = MarkerCluster(name="Points").add_to(m)

                # Add markers
                for _, row in filtered_gdf.iterrows():
                    popup_content = (
                        f"<b>Event Type:</b> {row['event_type']}<br>"
                        f"<b>Date:</b> {row['event_date']}<br>"
                        f"<b>Region:</b> {row['admin2']}<br>"
                        f"<b>Location:</b> {row['location']}<br>"
                        f"<b>Source:</b> {row['source']}"
                    )
                    
                    popup = folium.Popup(popup_content,min_width = 150, max_width=150)  # Adjust max_width as needed

                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=popup,
                        tooltip=row['event_type'], icon=folium.Icon(color='red',icon_color='white')
                    ).add_to(marker_cluster)

                # HeatMap layer
                heat_data = [[row['latitude'], row['longitude']] for _, row in filtered_gdf.iterrows()]

                HeatMap(heat_data, name="Heatmap").add_to(m)

                # Auto foucs
                bounds = [[row['latitude'], row['longitude']] for _, row in filtered_gdf.iterrows()]
                if bounds:
                    m.fit_bounds(bounds)

                # LayerControl
                folium.LayerControl(collapsed=True).add_to(m)

                st.components.v1.html(m._repr_html_(), height=294, scrolling=False)#height=340
            else:
                st.warning("Invalid map coordinates.")
        else:
            st.warning("No data available for the selected filters.")


    # Line chart for events by year
    with col_plot:
        # st.subheader("Events by Year")
        year_counts = filtered_gdf['year'].astype(int).value_counts().sort_index()
        fig2 = px.line(year_counts, labels={'year': 'Year', 'value': 'Number of Events'})
        fig2.update_xaxes(type='category')
        fig2.update_traces(mode='markers+lines', marker=dict(size=10))
        fig2.update_layout(showlegend=False)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, height=300, use_container_width=True)#height=346


    # Create two columns for the charts
    # col3 = st.columns(1, gap="small") 
    bar_plot__container = st.container()
    # Group data by admin1 and event_type
    grouped_gdf = filtered_gdf.groupby(['admin1', 'event_type']).size().reset_index(name='count')
    grouped_gdf = grouped_gdf.sort_values(by='count', ascending=False)

    # Horizontal bar chart for events by admin1
    with bar_plot__container:
        custom_colors = {
            "Battle":"#8B0000",        # Orange
            "Violence Against Civilians": "#ff7f0e",  # Red
            "Explosion/Remote Violence": "#9467bd",  # Purple
        }
        fig1 = px.bar(
            grouped_gdf,
            orientation='h',  
            y='admin1',      
            x='count',   
            color='event_type', 
            labels={'admin1': 'Administrative Region', 'count': 'Number of Events', 'event_type': 'Event Type'},
            barmode='stack', 
            color_discrete_map=custom_colors
        )

        # Update layout for better readability
        fig1.update_traces(textposition='inside')  # Display count values inside the bars
        fig1.update_layout(
            legend_title="Event Type",
            showlegend=True,
            # xaxis_title="Number of Events",
            # yaxis_title="Administrative Region"
        )
        fig1.update_layout(height=285)
        st.plotly_chart(fig1, height=285, use_container_width=True)#height=346



# Deepseek Chatbox
with tab2:

    def clean_response(text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() 
        for word in text:
            yield word 
            time.sleep(0.005)

    CHAT_URL = "http://127.0.0.1:8000/api/chat" 
    def get_chat_response(prompt, API_URL=CHAT_URL):
        """Send the prompt to FastAPI and return the response."""
        try:
            response = requests.post(API_URL, json={"prompt": prompt}, stream=False)
        
            if response.status_code == 200:
                raw_response = response.json().get("response", "Error: No response received.")
                return clean_response(raw_response)
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            raise Exception(f"Error: {str(e)}")  


    # st.header(f"WarChat with Deepseek")
    st.markdown(
    '<span style="font-size:2em; color:red;">WarChat</span>'
    '<span style="font-size:2em; color:white;"> with Deepseek</span>',
    unsafe_allow_html=True)
  
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # reset_chat()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt:= st.chat_input("Ask about the ongoing war in Sudan"):

        st.session_state.messages.append({"role": "user", "content": prompt})

        # User message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Model response
        with st.chat_message("assistant"):
    
            response = st.write_stream(get_chat_response(prompt))#get_chat_response(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []  # Reset messages
        st.rerun()  


st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f8f9fa;
            text-align: center;
            padding: 8px;
            font-size: 14px;
            font-weight: bold;
            color: #333;
        }
    </style>
    <div class="footer">
        Data Source: <a href="https://acleddata.com" target="_blank">ACLED</a> | © 2025 Wathela Alhassan    </div>
""", unsafe_allow_html=True)

