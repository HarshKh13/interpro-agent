from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import Column, Table, create_engine, ForeignKey, String, Integer, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
import os
import requests
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer

load_dotenv()
database_url = os.getenv("DATABASE_URL")
blast_server = os.getenv("BLAST_SERVER")


engine = create_engine(database_url)

class Base(DeclarativeBase):
    pass

class GoTerms(Base):
    __tablename__ = "go_terms"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    entities: Mapped[list["Entity"]] = relationship(secondary="entity_to_go_term_map",back_populates="go_terms")
    proteins: Mapped[list["Protein"]] = relationship(secondary="protein_to_go_term_map",back_populates="go_terms")
    embedding = mapped_column(Vector(384))
    
class Literature(Base):
    __tablename__ = "literature"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    url: Mapped[Optional[str]] = mapped_column(String,nullable=True)
    entities: Mapped[list["Entity"]] = relationship(secondary="entity_to_literature_map",back_populates="literatures")

class Entity(Base):
    __tablename__ = "entity"

    accession: Mapped[str] = mapped_column(String, primary_key=True)
    type: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(String,nullable=True)
    go_terms: Mapped[list[GoTerms]] = relationship(secondary="entity_to_go_term_map",back_populates="entities")
    literatures: Mapped[list[Literature]] = relationship(secondary="entity_to_literature_map",back_populates="entities")
    proteins: Mapped[list["Protein"]] = relationship(secondary="protein_to_entity_map",back_populates="entities")

class Organism(Base):
    __tablename__ = "organism"

    taxon_id: Mapped[str] = mapped_column(String,primary_key=True)
    scientific_name: Mapped[str] = mapped_column(String)
    full_name: Mapped[str] = mapped_column(String)
    proteins: Mapped[list["Protein"]] = relationship(back_populates="organism")

class Protein(Base):
    __tablename__ = "protein"

    accession: Mapped[str] = mapped_column(String,primary_key=True)
    taxon_id: Mapped[str] = mapped_column(ForeignKey("organism.taxon_id"))
    name: Mapped[Optional[str]] = mapped_column(String,nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String,nullable=True)
    length: Mapped[Optional[int]] = mapped_column(Integer)
    sequence: Mapped[Optional[str]] = mapped_column(String)
    proteome: Mapped[Optional[str]] = mapped_column(String,nullable=True)
    gene: Mapped[Optional[str]] = mapped_column(String)
    go_terms: Mapped[list[GoTerms]] = relationship(secondary="protein_to_go_term_map",back_populates="proteins")
    entities: Mapped[list[Entity]] = relationship(secondary="protein_to_entity_map",back_populates="proteins")
    organism: Mapped[Organism] = relationship(back_populates="proteins")


entity_to_go_term_map = Table(
    "entity_to_go_term_map",
    Base.metadata,
    Column("entity_accession", ForeignKey("entity.accession"), primary_key=True),
    Column("go_id", ForeignKey("go_terms.id"), primary_key=True)
)

entity_to_literature_map = Table(
    "entity_to_literature_map",
    Base.metadata,
    Column("entity_accession", ForeignKey("entity.accession"), primary_key=True),
    Column("literature_id", ForeignKey("literature.id"), primary_key=True)
)

protein_to_go_term_map = Table(
    "protein_to_go_term_map",
    Base.metadata,
    Column("protein_accession", ForeignKey("protein.accession"), primary_key=True),
    Column("go_id", ForeignKey("go_terms.id"), primary_key=True)
)

protein_to_entity_map = Table(
    "protein_to_entity_map",
    Base.metadata,
    Column("protein_accession", ForeignKey("protein.accession"), primary_key=True),
    Column("entity_accession", ForeignKey("entity.accession"), primary_key=True)
)


base_model = ChatOpenAI(
    openai_api_base="https://api.ai.it.ufl.edu",
    model = "gpt-oss-120b"
)

class ProteinState(TypedDict):
    accession: str
    taxon_id: str
    name: str
    description: str
    length: str
    sequence: str
    proteome: str
    gene: str
    go_terms: list[str]
    entities: list[str] 


class BlastHspState(TypedDict):
    bit_score: float
    score: float
    evalue: float
    identity: float
    positive: float
    query_from: int
    query_to: int
    hit_from: int
    hit_to: int
    align_length: int
    gaps: int
    query_sequence: str
    hit_sequence: str
    midline: str

class BlastHitState(TypedDict):
    accession: str
    length: int
    hsps: list[BlastHspState]

class GoTermState(TypedDict):
    id: str
    name: str
    category: str
    proteins: list[ProteinState]

class LiteratureState(TypedDict):
    id: str
    title: str
    url: str

class EntityState(TypedDict):
    accession: str
    type: str
    name: str
    description: str
    literatures: list[LiteratureState]
    proteins: list[ProteinState]

class InputQueryState(TypedDict):
    accession: Optional[str]
    gene: Optional[str]
    sequence: Optional[str]
    go_terms: Optional[list[str]]
    go_term_function: Optional[str]

class GraphState(TypedDict):
    query: str
    inputs: InputQueryState
    query_protein: ProteinState
    go_terms: list[GoTermState]
    blast_result: list[BlastHitState]
    entities: list[EntityState]
    tool_node: str


def get_protein_data(state: GraphState):
    """Find proteins by their attributes"""

    accession = state.get("inputs").get("accession")

    with Session(engine) as session:
        protein = session.query(Protein).filter(Protein.accession == accession).first()

        return {
            "query_protein": {
                "accession": protein.accession,
                "taxon_id": protein.taxon_id,
                "name": protein.name,
                "description": protein.description,
                "length": protein.length,
                "sequence": protein.sequence,
                "proteome": protein.proteome,
                "gene": protein.gene,
                "go_terms": [go_term.id for go_term in protein.go_terms],
                "entities": [entity.accession for entity in protein.entities]
            }
        }

def get_proteins_for_go_terms(state: GraphState):
    """Get all proteins associated with the given GO term IDs."""

    with Session(engine) as session:
        go_terms = session.query(GoTerms).filter(GoTerms.id.in_(state.get("inputs").get("go_terms"))).all()

        return {
            "go_terms": [
                {
                    "id": go_term.id,
                    "name": go_term.name,
                    "category": go_term.category,
                    "proteins": [
                        {
                            "accession": protein.accession,
                            "name": protein.name,
                            "description": protein.description,
                        }
                        for protein in go_term.proteins
                    ]
                }
                for go_term in go_terms
            ]
        }
    

def blast_search(state: GraphState):
    """Perform blast search for a protein"""

    blast_endpoint = f"{blast_server}/blast"

    query_sequence = state.get("inputs").get("sequence")
    if state.get("inputs").get("sequence") is None:
        with Session(engine) as session:
            protein = session.query(Protein).filter(Protein.accession == state.get("inputs").get("accession")).first()
            query_sequence = protein.sequence

    payload = {
        "organism": "arabidopsis", 
        "max_target_sequences":5,
        "query_sequence": query_sequence
    }

    response = requests.post(blast_endpoint, json=payload)
    result = response.json()

    return {
        "blast_result": result.get("blast_result")
    }

def get_proteins_for_go_term_functions(state: GraphState):
    go_term_function = state.get("inputs").get("go_term_function")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    go_term_function_embedding = model.encode(go_term_function).tolist()

    with Session(engine) as session:
        go_terms = (
            session.query(GoTerms,GoTerms.embedding.cosine_distance(go_term_function_embedding).label("distance")
            ).filter(GoTerms.embedding.cosine_distance(go_term_function_embedding) < 0.5)
            .order_by(GoTerms.embedding.cosine_distance(go_term_function_embedding)
            ).limit(10).all()
        )

        return {
            "go_terms": [
                {
                    "id": go_term.id,
                    "name": go_term.name,
                    "category": go_term.category,
                    "proteins": [
                        {
                            "accession": protein.accession,
                            "name": protein.name,
                            "description": protein.description,
                        }
                        for protein in go_term.proteins[:10]
                    ]
                }
                for go_term,distance in go_terms
            ]
        }



def input_extraction(state: GraphState):
    query = state.get("query")

    class Inputs(BaseModel):
        accession: Optional[str]
        gene: Optional[str]
        sequence: Optional[str]
        go_terms: Optional[list[str]]
        go_term_function: Optional[str]


    input_extraction_model = base_model.with_structured_output(Inputs)
    system_message = """You are a bioinformatics input extraction system.Extract the biological entities and identifiers explicitly mentioned
        or clearly expressed in the user's query.

        Return the following fields:

        1. accession
        - Protein accession identifiers.
        - Examples: A0A067E1Q0, A0A067E1J8

        2. gene
        - Gene names or gene symbols.

        3. sequence
        - Amino-acid or nucleotide sequences.

        4. go_terms
        - Explicit Gene Ontology identifiers only.
        - Examples: GO:0006096, GO:0006915, GO:0005524

        5. go_term_functions
        - Natural-language biological concepts that can correspond to
            Gene Ontology biological processes, molecular functions,
            or cellular components.
        - Extract every distinct biological concept from the query.

        Examples:

        User:
        "Which proteins are involved in glycolysis"

        go_term_functions: "glycolysis"

        User:
        "Which proteins are involved in the glucose pathway"
        go_term_function: "glucose pathway"

        User:
        "Find proteins involved in glucose metabolism"
        go_term_function: "glucose metabolism"

        User:
        "Find proteins with GO:0006096"
        go_terms: ["GO:0006096"]

        Only extract information relevant to the fields above.
        Do not generate information that is not present in the user's query.
        """
    
    output = input_extraction_model.invoke(
        [SystemMessage(content=system_message),
        HumanMessage(content=state.get("query"))]
    )

    print(output.go_term_function)

    return{
        "inputs": {
            "accession": output.accession,
            "gene": output.gene,
            "sequence": output.sequence,
            "go_terms": output.go_terms,
            "go_term_function": output.go_term_function
        }
    }

def router_node(state: GraphState):
    inputs = state.get("inputs", {})

    accession = inputs.get("accession")
    gene = inputs.get("gene")
    sequence = inputs.get("sequence")
    go_terms = inputs.get("go_terms")
    go_term_functions = inputs.get("go_term_function")

    if accession or gene:
        return {
            "tool_node": "get_protein_data"
        }

    if sequence:
        return {
            "tool_node": "blast_search"
        }

    if go_terms:
        return {
            "tool_node": "get_proteins_for_go_terms"
        }

    if go_term_functions:
        return {
            "tool_node": "get_proteins_for_go_term_functions"
        }

    return {
        "tool_node": None
    }

def route_next(state: GraphState):
    return state.get("tool_node")


graph_builder = StateGraph(GraphState)
graph_builder.add_node("input_extraction", input_extraction)
graph_builder.add_node("router_node", router_node)
graph_builder.add_node("get_protein_data", get_protein_data)
graph_builder.add_node("get_proteins_for_go_terms", get_proteins_for_go_terms)
graph_builder.add_node("get_proteins_for_go_term_functions", get_proteins_for_go_term_functions)
graph_builder.add_node("blast_search", blast_search)

graph_builder.add_edge(START, "input_extraction")
graph_builder.add_edge("input_extraction", "router_node")
graph_builder.add_conditional_edges("router_node", route_next)
graph_builder.add_edge("get_protein_data", END)
graph_builder.add_edge("get_proteins_for_go_terms", END)
graph_builder.add_edge("blast_search", END)
graph_builder.add_edge("get_proteins_for_go_term_functions", END)
graph = graph_builder.compile()

# initial_state = {
#     "query": "Which proteins are involved in ATP binding"
# }

# final_state = graph.invoke(initial_state)
# print(len(final_state.get("go_terms")))
# for go_term in final_state.get("go_terms"):
#     print(len(go_term.get("proteins")))