from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text, Column, Integer, Text, select
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from pydantic import BaseModel
import ollama
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
import os



load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

app = FastAPI()

engine = create_engine(DATABASE_URL)

@app.get("/api/events-json")
async def get_geojson():
    try:
        # Connect to the database and fetch the data
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT row_to_json(fc)
                    FROM (
                        SELECT 'FeatureCollection' as type, array_to_json(array_agg(f)) as features
                        FROM (
                            SELECT 'Feature' as type,
                                lg.event_date,
                                lg.year,
                                lg.event_type,
                                lg.actor1,
                                lg.actor2,
                                lg.admin1,
                                lg.admin2,
                                lg.location,
                                lg.fatalities,
                                lg.source,
                                lg.latitude, lg.longitude
                            FROM events lg
                        ) f
                    ) fc;
                """)
            )
            # Fetch the first row from the result
            geojson_data = result.fetchone()[0]

            return geojson_data

    except Exception as e:
        raise HTTPException(status_code=500, detail="Database query error: " + str(e))
        


PGV_DB_USER = os.getenv("PGV_DB_USER")
PGV_DB_PASSWORD = os.getenv("PGV_DB_PASSWORD")
PGV_DB_NAME = os.getenv("PGV_DB_NAME")
PGV_DB_HOST = os.getenv("PGV_DB_HOST")
PGV_DB_PORT = os.getenv("PGV_DB_PORT")

PGV_DATABASE_URL = f"postgresql://{PGV_DB_USER}:{PGV_DB_PASSWORD}@{PGV_DB_HOST}:{PGV_DB_PORT}/{PGV_DB_NAME}"
PGV_engine = create_engine(PGV_DATABASE_URL)

Base = declarative_base()


class KnowledgeBase(Base):
    __tablename__ = "warchat"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_col = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=False) 


Session = sessionmaker(bind=PGV_engine)
session = Session()


class RAG(KnowledgeBase):
    def __init__(self, request, session):
        self.request = request
        self.system_prompt = (
        "You are an expert on conflict events in Sudan, with a deep understanding of the ongoing war that began on April 15, 2023. "
        "Your task is to analyze and interpret conflict-related data, providing accurate, context-aware responses based on the available information. "
        "Ensure that your answers are precise, fact-based, and relevant to the ongoing situation in Sudan. "
        "Use the provided context to support your response, prioritizing clarity and objectivity.")
        self.session = session
        
    def embed_text(self, text):
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    
    def search_similar(self, top_k):
        query_embedding = self.embed_text(self.request.prompt)
        stmt = select(KnowledgeBase).order_by(
            KnowledgeBase.embedding.cosine_distance(query_embedding)
        ).limit(top_k)
        results = self.session.execute(stmt).scalars().all()
        return [r.text_col for r in results]
    
    def combine_system_prompt(self, top_k=10):
        context = self.search_similar(top_k)
        return f"{self.system_prompt}\n\nContext: {context}\n\nUser: {self.request.prompt}\nAssistant:"


class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str


@app.post("/api/chat", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    """Process a query by embedding it, retrieving knowledge, and generating a response."""
    

    # query_embedding = ollama.embeddings("nomic-embed-text", request.prompt)["embedding"]
    # with PGV_engine.connect() as conn:
    #     result = conn.execute(
    #         text("""
    #             SELECT text_col
    #             FROM warchat
    #             ORDER BY embedding <=> CAST(:query_embedding AS vector(768))
    #             LIMIT 5;
    #         """),
    #         {"query_embedding": query_embedding}
    #     )
    #     context = " ".join([row[0] for row in result.fetchall()])
    # print(context)

    
    input_text = RAG(request=request, session=session).combine_system_prompt()
    response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": input_text}])


    return QueryResponse(response=response.choices[0].message.content)#QueryResponse(response=response["message"]["content"])




