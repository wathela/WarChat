import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, Text, Date, Float, text, select
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
import ollama
from pyspark.sql import SparkSession
# from airflow import DAG
# from airflow.operators.bash import BashOperator
# from airflow.operators.python_operator import PythonOperator
# from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
import os
import gc
import re




# Event table schema 
PG_Base = declarative_base()

class Event(PG_Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_date = Column(Date)
    year = Column(Integer)
    event_type = Column(String)
    actor1 = Column(String)
    actor2 = Column(String)
    admin1 = Column(String)
    admin2 = Column(String)
    location = Column(String)
    fatalities = Column(Integer)
    source = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)

# Embedding table schema
PGV_Base = declarative_base()

class KnowledgeBase(PGV_Base):
    __tablename__ = "warchat"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_col = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=False)
def get_data(api_key, email):
    country = "sudan"
    limit = "0" # Get all relevant data
    url = f"https://api.acleddata.com/acled/read/?key={api_key}&email={email}&country={country}&limit={limit}"

    response = requests.get(url)
    if response.status_code == 200:

        data = response.json()['data']
        
        if response.status_code == 200:
            data = response.json()['data'] 
            df = pd.DataFrame(data) 
            print("Data successfully retrieved and loaded into DataFrame!")

            sub_df = df.loc[:, ['event_date', 'year', 'event_type', 'actor1', 'actor2','admin1','admin2', 'location', 'fatalities','source', 'latitude', 'longitude']].dropna()
            sub_df = sub_df[(sub_df['event_type'].isin(['Violence against civilians', 'Battles', 'Explosions/Remote violence'])) & (pd.to_datetime(sub_df['event_date']) >= "2023-04-14")]
            del data, df
            gc.collect()

            return sub_df
        
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")


def clean_actor_name(actor):
    if not isinstance(actor, str):
        return actor
    actor = re.sub(r".*?:", "", actor).strip()  
    actor = re.sub(r"\(\d{4,}-?\)", "", actor).strip() 
    return actor
   

def insert_df_into_postgres(df, engine, schema, if_exists):
   
    df.to_sql(name='events', con=engine, schema=schema, if_exists=if_exists, index=False) 
    del df
    gc.collect()
    print("Data successfully inserted into PostgreSQL!")

actor_list = [
    "Rapid Support Forces", "Darfur Communal Militia (Sudan)", "Military Forces of Sudan (2019-)", 
    "Misseriya Ethnic Militia (Sudan)", "ESLF: Eastern Sudan Liberation Forces",
    "SLM/A-Minnawi: Sudan Liberation Movement/Army (Minnawi Faction)",
    "Military Forces of Sudan (2019-) Military Intelligence Service", "Sudan Shield Forces", 
    "Military Forces of Sudan (2019-) Military Police",
    "SPLM-N-Abdelaziz: Sudan People's Liberation Movement (North) (Abdelaziz al-Hilu faction)",
    "Al Baraa Ibn Malik Brigade", "Darfur Arab Militia (Sudan)", "RAC: Revolutionary Awakening Council",
    "West Kordofan Communal Militia (Sudan)", "Darfur Joint Forces/JSAMF: Joint Force of Armed Struggle Movement",
    "Salamat Ethnic Militia (Sudan)", "South Kordofan Communal Militia (Sudan)", 
    "Police Forces of Sudan (2019-) General Intelligence Service", "Wad Al Nura Communal Militia (Sudan)", 
    "Qaws: Popular Resistance Movement to Support the Armed Forces", "Unidentified Armed Group (Chad)", 
    "Misseriya Jebel Ethnic Militia (Sudan)", "Zaghawa Ethnic Militia (Sudan)", 
    "Military Forces of Egypt (2014-)", "Police Forces of Sudan (2019-)", 
    "North Kordofan Communal Militia (Sudan)", "Awlad Gomri", "El Bazaa Ethnic Militia (Sudan)", 
    "Reserve Forces Eagles Brigade", "Sudanese Popular Resistance Factions", 
    "Al Zindiya Ethnic Militia (Sudan)", "Kordofan Arab Militia (Sudan)", 
    "Kambo Tayba Communal Militia (Sudan)", "Nuer Ethnic Militia (Sudan)", "Najah Ethnic Militia (Sudan)", 
    "SLM/A-Transitional Council: Sudan Liberation Movement/Army (Transitional Council Faction)", 
    "SLMJ-TH: Sudan Liberation Movement for Justice (Taher Hajar Faction)", "Ngok Clan Militia (Sudan)", 
    "Zankaha Al Khawalda Communal Militia (Sudan)", "Al Jumueia Communal Militia (Sudan)", 
    "Hawazmah-Dar Ali-Sub-Clan Militia (Sudan)", "Berti Ethnic Militia (Sudan)", 
    "SLM/A-Tambor: Sudan Liberation Movement/Army (Mustafa Tambor Faction)", 
    "Military Forces of Chad (2021-)", "JEM-Gibiril Ibrahim: Justice and Equality Movement", 
    "Shangil Tobay Communal Militia (Sudan)", "Blue Nile Communal Militia (Sudan)", 
    "SLM/A-Transitional Council-Salah Rassas: Sudan Liberation Movement/Army (Transitional Council-Salah Rassas Faction)", 
    "Military Forces of South Sudan (2011-)", "Shugara Forces", "Military Forces of Sudan (2019-) Abu Tira", 
    "Awlad Rashid Ethnic Militia (Sudan)", "Mahriya Clan Militia (Sudan)", "Nuba Ethnic Militia (Sudan)", 
    "Rizeigat Ethnic Militia (Sudan)", "Al Soriba Communal Militia (Sudan)", "Al-Falata Ethnic Militia (Sudan)", 
    "Wad Kibeish Communal Militia (Sudan)", "Al Jukhaisat Ethnic Militia (Sudan)", 
    "Habbaniyah Ethnic Militia (Sudan)", "Baggara Ethnic Militia (Sudan)", "Dar Hamid Ethnic Militia (Sudan)", 
    "Jabal Moya Communal Militia (Sudan)", "Batahin Ethnic Militia (Sudan)", "Wad Medani Communal Militia (Sudan)", 
    "Aura Ethnic Militia (Sudan)", "El Fula Communal Militia (Sudan)", "Hawazmah Clan Militia (Sudan)", 
    "GSLF: Gathering of Sudan Liberation Forces", "Hamar Ethnic Militia (Sudan)", "Rabak Communal Militia (Sudan)", 
    "Ingessana Ethnic Militia (Sudan)", "Kawahala Ethnic Militia (Sudan)", 
    "Military Forces of Ukraine (2019-)", "Beni Halba Ethnic Militia (Sudan)", "Kababeesh Ethnic Militia (Sudan)", 
    "Bul Clan Militia (South Sudan)", "Ziyadiyah Ethnic Militia (Sudan)", "Awlad Mansour Clan Militia (Sudan)", 
    "Bendasi Communal Militia (Sudan)", "Hawita Ethnic Militia (Sudan)", 
    "SLM/A-Peace and Development: Sudan Liberation Movement/Army (Peace and Development Faction)", 
    "Masalit Ethnic Militia (Sudan)", "Dinka Ethnic Militia (Sudan)", "Sudanese Alliance Movement", 
    "SLM/A: Sudan Liberation Movement/Army", "Hausa Ethnic Militia (Sudan)", "Zalingei Communal Militia (Sudan)"
]

def get_and_ingest_data(API_KEY, EMAIL, actor_list, engine, schema):

    try:
        df = get_data(API_KEY, EMAIL)
        
        filtered_df = df[df["actor1"].isin(actor_list) | df["actor2"].isin(actor_list)].copy()

        filtered_df["actor1"] = filtered_df["actor1"].apply(clean_actor_name)
        filtered_df["actor2"] = filtered_df["actor2"].apply(clean_actor_name)
        
        insert_df_into_postgres(filtered_df, engine, schema, "replace")

        return filtered_df
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

def embed_text(text):
    """Generate embeddings using Ollama."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def embed_dataframe(df, chunk_size, num_chunks, session):
    """Convert df to embeddings and store."""

    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]  # Slice 300-row chunk

        # Generate text descriptions
        # text_data = chunk.apply(lambda row: 
        #     f"On {row['event_date']}, a {row['event_type']} event involving {row['actor1']} and {row['actor2']} "
        #     f"occurred in {row['location']}, {row['admin1']}, resulting in {row['fatalities']} fatalities. "
        #     f"The source of this information is {row['source']}.", axis=1).tolist()
        text_data = chunk.apply(lambda row: 
            f"⚠️DATE: {row['event_date']}⚠️ - A {row['event_type']} event occurred at 📍LOCATION: {row['location']}, {row['admin1']}📍. "
            f"Involved parties: {row['actor1']} vs {row['actor2']}. Fatalities: {row['fatalities']}. "
            f"Source: {row['source']}.", axis=1).tolist()
        
        text_data = " | ".join(text_data) 
        embedding_vector = embed_text(text_data)

        new_entry = KnowledgeBase(text_col=text_data, 
                                embedding=embedding_vector,)
        session.add(new_entry)

    session.commit()
    print("Embeddings inserted successfully.")

def emdeb_insert_data(filtered_df, PGV_engine, replace=True):
    
    if replace:
        with PGV_engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS warchat CASCADE;"))
            conn.commit()
            print("Table 'warchat' dropped successfully.")

    # Create the table in the database
    PGV_Base.metadata.create_all(PGV_engine)
    Session = sessionmaker(bind=PGV_engine)
    session = Session()
    print("Table 'warchat' has been created successfully.")

    chunk_size = 1
    num_chunks = (len(filtered_df) + chunk_size - 1) // chunk_size
    embed_dataframe(filtered_df, chunk_size, num_chunks, session)

def run_spark_job(API_KEY, EMAIL, actor_list, PG_engine, PGV_engine):
    spark = SparkSession.builder \
        .appName("AirflowSparkJob") \
        .getOrCreate()
        #.config("spark.driver.bindAddress","127.0.0.1") \
        #   .config("spark.executor.memory", "2g") \
        # .master("spark://spark-master:7077") \
        
    
    filtered_df = get_and_ingest_data(API_KEY, EMAIL, actor_list, PG_engine, 'public')
   
    emdeb_insert_data(filtered_df, PGV_engine, replace=True)
    print("Spark Job Completed")
    spark.stop()


if __name__ == "__main__":

    load_dotenv()

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    API_KEY = os.getenv("API_KEY")
    EMAIL = os.getenv("EMAIL")
    PGV_DB_USER = os.getenv("PGV_DB_USER")
    PGV_DB_PASSWORD = os.getenv("PGV_DB_PASSWORD")
    PGV_DB_NAME = os.getenv("PGV_DB_NAME")
    PGV_DB_HOST = os.getenv("PGV_DB_HOST")
    PGV_DB_PORT = os.getenv("PGV_DB_PORT")

    PGV_DATABASE_URL = f"postgresql+psycopg://{PGV_DB_USER}:{PGV_DB_PASSWORD}@{PGV_DB_HOST}:{PGV_DB_PORT}/{PGV_DB_NAME}"
    PG_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    PG_engine = create_engine(PG_DATABASE_URL)
    PGV_engine = create_engine(PGV_DATABASE_URL)

    run_spark_job(API_KEY, EMAIL, actor_list, PG_engine, PGV_engine)


     










