from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict, Annotated, Optional
from langchain_core.messages import ToolMessage, AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import Column, Table, create_engine, ForeignKey, String, Integer, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
import os
import requests
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
import subprocess
import json
from sqlalchemy import or_

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import NotRequired
import tempfile
import shutil
from pathlib import Path


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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


def merge_by_key(key: str):
    def merge(existing: list, incoming: list) -> list:
        merged = {item[key]: item for item in existing}

        for item in incoming:
            merged[item[key]] = item

        return list(merged.values())

    return merge


merge_proteins = merge_by_key("accession")
merge_go_terms = merge_by_key("id")
merge_entities = merge_by_key("accession")

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

class ProteinSummaryState(TypedDict):
    accession: str
    name: str
    description: str

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
    proteins: list[ProteinSummaryState]

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
    proteins: list[ProteinSummaryState]

class InputQueryState(TypedDict):
    accession: Optional[str]
    gene: Optional[str]
    sequence: Optional[str]
    go_terms: Optional[list[str]]
    go_term_function: Optional[str]
    entity_accession: Optional[str]

class GraphState(TypedDict):
    query: str
    inputs: NotRequired[InputQueryState]
    query_protein: NotRequired[ProteinState]
    proteins: Annotated[list[ProteinState],merge_proteins]
    go_terms: Annotated[list[GoTermState],merge_go_terms]
    blast_result: NotRequired[list[BlastHitState]]
    entities: Annotated[list[EntityState],merge_entities]
    tool_node: str
    messages: Annotated[list[AnyMessage], add_messages]
    ortholog_results: NotRequired[list[dict]]

def serialize_protein(protein: Protein) -> ProteinState:
    return {
        "accession": protein.accession,
        "taxon_id": protein.taxon_id,
        "name": protein.name or "",
        "description": protein.description or "",
        "length": str(protein.length) if protein.length is not None else "",
        "sequence": protein.sequence or "",
        "proteome": protein.proteome or "",
        "gene": protein.gene or "",
        "go_terms": [
            go_term.id for go_term in protein.go_terms
        ],
        "entities": [
            entity.accession for entity in protein.entities
        ]
    }

def serialize_go_term(go_term: GoTerms) -> GoTermState:
    return {
        "id": go_term.id,
        "name": go_term.name,
        "category": go_term.category,
        "proteins": [
            {
                "accession": protein.accession,
                "name": protein.name or "",
                "description": protein.description or "",
            }
            for protein in go_term.proteins[:10]
        ]
    }

def serialize_entity(entity: Entity) -> EntityState:
    return {
        "accession": entity.accession,
        "type": entity.type,
        "name": entity.name,
        "description": entity.description or "",
        "literatures": [
            {
                "id": literature.id,
                "title": literature.title,
                "url": literature.url or ""
            }
            for literature in entity.literatures
        ],
        "proteins": [
            {
                "accession": protein.accession,
                "name": protein.name or "",
                "description": protein.description or "",
            }
            for protein in entity.proteins[:10]
        ]
    }

@tool
def get_protein(accession: str, runtime: ToolRuntime) -> Command:
    """
    Retrieve detailed information about a protein using its accession number.

    Use this tool when the user asks for information about a specific protein,
    including its name, gene, sequence, organism, GO annotations, or
    associated biological entities.

    Args:
        accession: Unique accession number identifying the protein.

    The retrieved protein is stored in GraphState.query_protein.
    """

    with Session(engine) as session:
        protein = session.query(Protein).filter(Protein.accession == accession).first()
        if protein is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Protein {accession} not found.",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )
        protein_data = serialize_protein(protein)
        
    return Command(
        update={
            "query_protein": protein_data,
            "messages": [
                ToolMessage(
                    content=f"Successfully loaded protein {accession}.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

@tool
def search_proteins(query: str, runtime: ToolRuntime, limit: int = 20) -> Command:
    """
    Search proteins using their gene name, protein name, or description.

    Use this tool for text-based searches, such as finding proteins
    whose names or descriptions contain 'kinase'.

    Do not use for GO functional annotation searches.
    Use GO term search tools for biological function queries.

    Args:
        query: Search keyword or phrase.
        limit: Maximum number of proteins to return.
    """

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    limit = max(1, min(limit, 100))

    with Session(engine) as session:
        protein_query = session.query(Protein).filter(
            or_(
                Protein.name.ilike(f"%{query}%"),
                Protein.gene.ilike(f"%{query}%"),
                Protein.description.ilike(f"%{query}%")
            )
        )

        if protein_query is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Proteins not found.",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

        proteins = (protein_query.order_by(Protein.accession).limit(limit).all())
        protein_data = [serialize_protein(protein) for protein in proteins]

    return Command(
        update={
            "proteins": protein_data,
            "messages": [
                ToolMessage(
                    content=(
                        f"Found {len(protein_data)} proteins "
                        f"matching '{query}'."
                    ),
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

@tool
def get_go_term(go_id: str, runtime: ToolRuntime) -> Command:
    """
    Retrieve a GO term using its unique GO identifier.

    Use this tool when the user asks for information about a
    specific GO term, including its name, category, and
    directly associated proteins.

    Args:
        go_id: GO term identifier, such as GO:0004672.

    Stores the retrieved GO term in GraphState.go_terms.
    """

    with Session(engine) as session:
        go_term = session.query(GoTerms).filter(GoTerms.id == go_id).first()

        if go_term is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"GO term {go_id} not found.",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

        go_term_data = serialize_go_term(go_term)

    return Command(
        update={
            "go_terms": [go_term_data],
            "messages": [
                ToolMessage(
                    content=f"Successfully retrieved GO term {go_id}.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

def generate_embedding(text: str) -> list[float]:
    return embedding_model.encode(text, normalize_embeddings=True).tolist()

@tool
def search_go_terms(query: str, runtime: ToolRuntime, limit: int = 10, similarity_threshold: float = 0.65) -> Command:
    """
    Search Gene Ontology (GO) terms using text matching
    and semantic vector similarity.

    Use this tool when the researcher describes a biological
    function or process without providing an exact GO ID.

    Examples:
    - kinase activity
    - plant immunity
    - photosynthesis
    - response to oxidative stress

    Args:
        query: Biological function or process to search for.
        limit: Maximum number of matching GO terms.
        similarity_threshold: Minimum cosine similarity
            for semantic matches, between 0 and 1.

    Stores matching GO terms in GraphState.go_terms.
    """

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    limit = max(1, min(limit, 50))
    query_embedding = generate_embedding(query)

    with Session(engine) as session:

        text_matches = (
            session.query(GoTerms).filter(GoTerms.name.ilike(f"%{query}%")).order_by(GoTerms.id).limit(limit).all())

        cosine_distance = GoTerms.embedding.cosine_distance(query_embedding)

        semantic_matches = (session.query(GoTerms).filter(GoTerms.embedding.isnot(None))
            .filter(cosine_distance <= 1 - similarity_threshold).order_by(cosine_distance).limit(limit).all())

        combined = {}

        for go_term in text_matches:
            combined[go_term.id] = go_term

        for go_term in semantic_matches:
            if go_term.id not in combined:
                combined[go_term.id] = go_term

        go_term_data = [
            serialize_go_term(go_term)
            for go_term in list(combined.values())[:limit]
        ]

    return Command(
        update={
            "go_terms": go_term_data,
            "messages": [
                ToolMessage(
                    content=(
                        f"Found {len(go_term_data)} GO terms "
                        f"related to '{query}'. "
                        f"GO IDs: {[go['id'] for go in go_term_data]}"
                    ),
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

@tool
def get_go_terms_for_protein(accession: str, runtime: ToolRuntime) -> Command:
    """
    Retrieve all GO terms directly associated with a protein.

    Use this tool when the researcher asks about the
    biological processes, molecular functions, or cellular
    components associated with a specific protein.

    Args:
        accession: Unique accession number of the protein.

    Stores the retrieved GO terms in GraphState.go_terms.
    """

    with Session(engine) as session:

        protein = (session.query(Protein).filter(Protein.accession == accession).first())

        if protein is None:
            return Command(
                update={
                    "go_terms": [],
                    "messages": [
                        ToolMessage(
                            content=f"Protein {accession} not found.",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

        go_term_data = [
            serialize_go_term(go_term)
            for go_term in protein.go_terms
        ]

    return Command(
        update={
            "go_terms": go_term_data,
            "messages": [
                ToolMessage(
                    content=(
                        f"Retrieved {len(go_term_data)} GO terms "
                        f"associated with protein {accession}. "
                        f"GO IDs: {[go['id'] for go in go_term_data]}"
                    ),
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

@tool
def get_proteins_for_go_term(go_id: str, runtime: ToolRuntime, limit: int = 50) -> Command:
    """
    Retrieve proteins directly associated with a GO term.

    Use this tool when the researcher asks for proteins
    associated with a specific GO identifier.

    Examples:
    - Find proteins associated with GO:0004672.
    - Find Citrus sinensis proteins associated with GO:0004672.

    Args:
        go_id: Unique GO identifier, such as GO:0004672.
        limit: Maximum number of proteins to retrieve (1-100).

    Stores the retrieved proteins in GraphState.proteins.
    """

    limit = max(1, min(limit, 100))

    with Session(engine) as session:
        go_term = session.query(GoTerms).filter(GoTerms.id == go_id).first()

        if go_term is None:
            return Command(
                update={
                    "proteins": [],
                    "messages": [
                        ToolMessage(
                            content=f"GO term {go_id} not found.",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

        protein_query = session.query(Protein).filter(Protein.go_terms.any(GoTerms.id == go_id))
        proteins = protein_query.order_by(Protein.accession).limit(limit).all()
        protein_data = [serialize_protein(protein) for protein in proteins]

    return Command(
        update={
            "proteins": protein_data,
            "messages": [
                ToolMessage(
                    content=(
                        f"Retrieved {len(protein_data)} proteins "
                        f"associated with GO term {go_id}"
                    ),
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

@tool
def get_entity(accession: str, runtime: ToolRuntime) -> Command:
    """
    Retrieve detailed information about a specific biological
    entity using its accession number.

    Use this tool when the researcher asks about an entity's
    name, type, description, associated proteins, or literature.

    Args:
        accession: Unique entity accession, such as IPR000719.

    Stores the retrieved entity in GraphState.entities.
    """

    with Session(engine) as session:
        entity = session.query(Entity).filter(Entity.accession == accession).first()

        if entity is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Entity {accession} not found.",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

        entity_data = serialize_entity(entity)

    return Command(
        update={
            "entities": [entity_data],
            "messages": [
                ToolMessage(
                    content=f"Successfully retrieved entity {accession}.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )


@tool
def search_entities(query: str, runtime: ToolRuntime, entity_type: str | None = None,limit: int = 20) -> Command:
    """
    Search biological entities by name or description.

    Use this tool when the researcher wants to find entities
    related to a biological concept but does not know their
    exact accession number.

    Optionally filter by entity type.

    Examples:
    - Find entities related to protein kinase domains.
    - Search entities associated with DNA binding.
    - Find domains related to phosphorylation.

    Args:
        query: Search phrase for entity names or descriptions.
        entity_type: Optional entity type, such as 'domain'.
        limit: Maximum number of matching entities (1-100).

    Stores matching entities in GraphState.entities.
    """
    limit = max(1, min(limit, 100))

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    with Session(engine) as session:
        entity_query = session.query(Entity).filter(or_(Entity.name.ilike(f"%{query}%"),Entity.description.ilike(f"%{query}%")))

        if entity_type is not None:
            entity_query = entity_query.filter(Entity.type == entity_type)

        entities = entity_query.order_by(Entity.accession).limit(limit).all()
        entity_data = [serialize_entity(entity) for entity in entities]

    return Command(
        update={
            "entities": entity_data,
            "messages": [
                ToolMessage(
                    content=(
                        f"Found {len(entity_data)} entities "
                        f"matching '{query}'."
                    ),
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

@tool
def get_entities_for_protein(accession: str, runtime: ToolRuntime) -> Command:
    """
    Retrieve all biological entities directly associated
    with a specific protein.

    Use this tool when the researcher asks about domains,
    families, or other entities linked to a protein.

    Args:
        accession: Unique accession number of the protein.

    Stores the retrieved entities in GraphState.entities.
    """

    with Session(engine) as session:
        protein = session.query(Protein).filter(Protein.accession == accession).first()

        if protein is None:
            return Command(
                update={
                    "entities": [],
                    "messages": [
                        ToolMessage(
                            content=f"Protein {accession} not found.",
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

        entity_data = [serialize_entity(entity) for entity in protein.entities]

    return Command(
        update={
            "entities": entity_data,
            "messages": [
                ToolMessage(
                    content=(
                        f"Retrieved {len(entity_data)} entities "
                        f"associated with protein {accession}. "
                    ),
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )


@tool
def get_proteins_for_entity(entity_accession: str, runtime: ToolRuntime, limit: int = 50) -> Command:
    """
    Retrieve proteins directly associated with a specific
    biological entity.

    Use this tool when the researcher asks for proteins
    associated with an entity, such as an InterPro domain,
    protein family, or another annotated biological entity.


    Examples:
    - Find proteins associated with entity IPR000719.
    - Find Citrus sinensis proteins linked to IPR000719.

    Args:
        entity_accession: Unique accession of the entity.
        limit: Maximum number of proteins to retrieve (1-100).

    Stores retrieved proteins in GraphState.proteins.
    """

    limit = max(1, min(limit, 100))

    with Session(engine) as session:
        entity = session.query(Entity).filter(Entity.accession == entity_accession).first()

        if entity is None:
            return Command(
                update={
                    "proteins": [],
                    "messages": [
                        ToolMessage(
                            content=(
                                f"Entity {entity_accession} not found."
                            ),
                            tool_call_id=runtime.tool_call_id
                        )
                    ]
                }
            )

        protein_query = session.query(Protein).filter(Protein.entities.any(Entity.accession == entity_accession))
        proteins = protein_query.order_by(Protein.accession).limit(limit).all()
        protein_data = [serialize_protein(protein) for protein in proteins]

    return Command(
        update={
            "proteins": protein_data,
            "messages": [
                ToolMessage(
                    content=(
                        f"Retrieved {len(protein_data)} proteins "
                        f"associated with entity {entity_accession}"
                    ),
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )


def _write_protein_fasta(proteins, fasta_path):
    """Write database protein sequences to a local FASTA file."""
    count = 0

    with open(fasta_path, "w") as handle:
        for protein in proteins:
            sequence = (protein.sequence or "").upper()
            sequence = "".join(sequence.split())

            if not sequence or not protein.accession:
                continue

            handle.write(f">{protein.accession}\n")
            for i in range(0, len(sequence), 60):
                handle.write(sequence[i:i + 60] + "\n")

            count += 1

    if count == 0:
        raise ValueError("No protein sequences available for BLAST.")

    return count


def _make_blast_database(fasta_path, db_prefix):
    """Build a local protein database using BLAST+."""
    subprocess.run(
        [
            "makeblastdb",
            "-in", str(fasta_path),
            "-dbtype", "prot",
            "-out", str(db_prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _run_local_blast(query_fasta, db_prefix, limit=10):
    """
    Return local BLASTP hits with identity, coverage,
    E-value, and bit score.
    """
    result = subprocess.run(
        [
            "blastp",
            "-query", str(query_fasta),
            "-db", str(db_prefix),
            "-outfmt",
            "6 qseqid sseqid pident length qlen slen evalue bitscore",
            "-max_target_seqs", str(limit),
            "-evalue", "10",
            "-num_threads", "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )

    hits = []

    for line in result.stdout.splitlines():
        fields = line.split("\t")

        if len(fields) != 8:
            continue

        query_id, subject_id = fields[0], fields[1]
        identity = float(fields[2])
        alignment_length = int(fields[3])
        query_length = int(fields[4])
        subject_length = int(fields[5])
        evalue = float(fields[6])
        bitscore = float(fields[7])

        hits.append({
            "query_accession": query_id,
            "accession": subject_id,
            "identity_percent": round(identity, 2),
            "query_coverage_percent": round(
                100 * alignment_length / query_length, 2
            ) if query_length else 0.0,
            "subject_coverage_percent": round(
                100 * alignment_length / subject_length, 2
            ) if subject_length else 0.0,
            "evalue": evalue,
            "bit_score": bitscore,
        })

    return hits


@tool
def analyze_orthologs(
    accession: str,
    target_taxon_id: str,
    runtime: ToolRuntime,
    limit: int = 10,
) -> Command:
    """
    Identify candidate orthologs for a protein in another
    organism using local BLASTP and reciprocal best hits.

    accession:
        Accession of the source protein in the local database.

    target_taxon_id:
        Taxonomy ID of the target organism already stored
        in the local organism table.

    limit:
        Number of top BLAST candidates to evaluate, from 1 to 50.

    No external APIs are used.
    """
    limit = max(1, min(limit, 50))

    if not shutil.which("blastp") or not shutil.which("makeblastdb"):
        raise RuntimeError(
            "BLAST+ is not installed or is missing from PATH. "
            "Install blastp and makeblastdb locally."
        )

    # Load both proteomes from the existing PostgreSQL database.
    with Session(engine) as session:
        source = (
            session.query(Protein)
            .filter(Protein.accession == accession)
            .first()
        )

        if source is None:
            raise ValueError(
                f"Source protein {accession} was not found."
            )

        if not source.sequence:
            raise ValueError(
                f"Protein {accession} has no stored sequence."
            )

        if source.taxon_id == target_taxon_id:
            raise ValueError(
                "Choose a target organism different from the source."
            )

        source_taxon_id = source.taxon_id
        source_sequence = source.sequence
        source_name = (
            session.query(Organism)
            .filter(Organism.taxon_id == source_taxon_id)
            .first()
        )

        target_organism = (
            session.query(Organism)
            .filter(Organism.taxon_id == target_taxon_id)
            .first()
        )

        if target_organism is None:
            raise ValueError(
                f"Target organism {target_taxon_id} is not in the database."
            )

        source_proteins = (
            session.query(Protein)
            .filter(Protein.taxon_id == source_taxon_id)
            .filter(Protein.sequence.isnot(None))
            .all()
        )

        target_proteins = (
            session.query(Protein)
            .filter(Protein.taxon_id == target_taxon_id)
            .filter(Protein.sequence.isnot(None))
            .all()
        )

        # Copy the ORM data needed after closing the session.
        source_rows = [
            (p.accession, p.sequence)
            for p in source_proteins
            if p.sequence
        ]
        target_rows = [
            (p.accession, p.sequence)
            for p in target_proteins
            if p.sequence
        ]

        source_organism_name = (
            source_name.scientific_name if source_name else source_taxon_id
        )
        target_organism_name = target_organism.scientific_name

    if not source_rows or not target_rows:
        raise ValueError(
            "A source or target proteome has no stored sequences."
        )

    # Use isolated temporary files. Nothing is uploaded externally.
    with tempfile.TemporaryDirectory(prefix="ortholog_analysis_") as tmp:
        workdir = Path(tmp)

        source_fasta = workdir / "source.fasta"
        target_fasta = workdir / "target.fasta"
        query_fasta = workdir / "query.fasta"
        candidate_fasta = workdir / "candidate.fasta"

        # Create FASTA files from the locally stored proteomes.
        _write_protein_fasta(
            [
                type("ProteinRecord", (), {
                    "accession": acc, "sequence": seq
                })()
                for acc, seq in source_rows
            ],
            source_fasta,
        )

        _write_protein_fasta(
            [
                type("ProteinRecord", (), {
                    "accession": acc, "sequence": seq
                })()
                for acc, seq in target_rows
            ],
            target_fasta,
        )

        source_db = workdir / "source_db"
        target_db = workdir / "target_db"

        _make_blast_database(source_fasta, source_db)
        _make_blast_database(target_fasta, target_db)

        # Forward search: source protein against target proteome.
        with open(query_fasta, "w") as handle:
            handle.write(f">{accession}\n{source_sequence}\n")

        forward_hits = _run_local_blast(
            query_fasta, target_db, limit=limit
        )

        target_sequences = {
            acc: seq for acc, seq in target_rows
        }

        source_accessions = {
            acc for acc, seq in source_rows
        }

        results = []

        # Reverse search each candidate against the source proteome.
        for hit in forward_hits:
            candidate_accession = hit["accession"]
            candidate_sequence = target_sequences.get(
                candidate_accession
            )

            if not candidate_sequence:
                continue

            with open(candidate_fasta, "w") as handle:
                handle.write(
                    f">{candidate_accession}\n"
                    f"{candidate_sequence}\n"
                )

            reverse_hits = _run_local_blast(
                candidate_fasta, source_db, limit=1
            )

            reverse_best = (
                reverse_hits[0] if reverse_hits else None
            )

            reciprocal_best_hit = (
                reverse_best is not None
                and reverse_best["accession"] == accession
            )

            results.append({
                **hit,
                "source_accession": accession,
                "source_taxon_id": source_taxon_id,
                "source_organism": source_organism_name,
                "target_taxon_id": target_taxon_id,
                "target_organism": target_organism_name,
                "reverse_best_accession": (
                    reverse_best["accession"]
                    if reverse_best else None
                ),
                "reciprocal_best_hit": reciprocal_best_hit,
                "orthology_status": (
                    "RBH candidate"
                    if reciprocal_best_hit
                    else "Similar protein; RBH not confirmed"
                ),
            })

    # Rank RBH candidates first, then by sequence evidence.
    results.sort(
        key=lambda item: (
            not item["reciprocal_best_hit"],
            item["evalue"],
            -item["bit_score"],
        )
    )

    summary = {
        "source_accession": accession,
        "source_organism": source_organism_name,
        "target_organism": target_organism_name,
        "target_taxon_id": target_taxon_id,
        "candidate_count": len(results),
        "rbh_count": sum(
            item["reciprocal_best_hit"] for item in results
        ),
        "results": results,
        "method": "Local BLASTP + reciprocal best-hit analysis",
        "external_api_used": False,
        "caveat": (
            "Reciprocal best hits are candidate orthologs, "
            "not definitive proof of orthology."
        ),
    }

    return Command(
        update={
            "ortholog_results": [summary],
            "messages": [
                ToolMessage(
                    content=json.dumps(summary, indent=2),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


base_model = ChatOpenAI(
    openai_api_base="https://api.ai.it.ufl.edu",
    model = "gpt-oss-120b"
)

tools = [
    get_protein,
    search_proteins,

    get_go_term,
    search_go_terms,
    get_go_terms_for_protein,
    get_proteins_for_go_term,

    get_entity,
    search_entities,
    get_entities_for_protein,
    get_proteins_for_entity,

    analyze_orthologs
]

llm_with_tools = base_model.bind_tools(tools, parallel_tool_calls=False)

SYSTEM_PROMPT = """
You are an expert biological research assistant.

You answer research questions using a PostgreSQL
database containing proteins, Gene Ontology terms,
biological entities, literature, and organisms.

You have access to database tools.

DATABASE RELATIONSHIPS

Protein -> GO Terms
Protein -> Entities
Protein -> Organism
Entity -> GO Terms
Entity -> Literature
Entity -> Proteins
GO Term -> Proteins

TOOL SELECTION RULES

1. Specific protein accession:
   Use get_protein.

2. Protein name, gene, or description:
   Use search_proteins.

3. Specific GO identifier:
   Use get_go_term or get_proteins_for_go_term.

4. Biological function without a GO identifier:
   Use search_go_terms.

5. GO terms associated with a protein:
   Use get_go_terms_for_protein.

6. Proteins associated with a biological function:
   First search_go_terms.
   Then inspect the matching GO identifiers.
   Then call get_proteins_for_go_term.

7. Specific entity accession:
   Use get_entity.

8. Entity name or description:
   Use search_entities.

9. Entities associated with a protein:
   Use get_entities_for_protein.

10. Proteins associated with an entity:
    Use get_proteins_for_entity.

11. Ortholog analysis across organisms:
    - Use analyze_orthologs when the user asks to find
      orthologs or corresponding proteins in another organism.
    - Use the exact source protein accession.
    - Obtain the target organism's verified taxon ID from
      the database; never guess it.
    - Report sequence identity, query coverage, E-value,
      reciprocal-best-hit status, and accession.
    - Distinguish RBH candidates from confirmed orthology.
    - This tool uses local BLAST+ and local database sequences.

MULTI-STEP WORKFLOW RULES

- Break complex questions into logical steps.
- Execute dependent steps in the correct order.
- Call one tool at a time.
- Use actual results from previous tools.
- Never invent accessions or GO identifiers.
- Do not call tools with guessed identifiers.
- If a search returns several candidates, select
  only candidates relevant to the user's request.
- Do not repeat the same tool call unnecessarily.
- Do not stop until the requested information has
  been retrieved, or a limitation prevents completion.
- Do not claim that a partial result is exhaustive.
- If the user specifies an organism, filter results
  by the protein's taxon_id before answering.
- For Citrus sinensis, taxon_id is 2711.
- Other organism names require verified taxonomy
  identifiers; do not guess them.

STATE RULES

Tool results are stored in shared GraphState.

The current state summary is provided below.

Use state data to decide the next tool call.

If the required data already exists in the state,
do not retrieve it again unnecessarily.

The state may contain results from multiple tools.

ANSWER RULES

- Base conclusions on retrieved data.
- Distinguish GO annotations from entity associations.
- Include relevant protein accessions and GO IDs.
- Clearly state when results are incomplete.
- Do not invent biological relationships.
- If the tools cannot establish a requested
  relationship, explain the limitation.
"""

def build_state_context(state: GraphState) -> str:

    proteins = state.get("proteins", [])
    go_terms = state.get("go_terms", [])
    entities = state.get("entities", [])

    query_protein = state.get("query_protein")

    context = {
        "query": state["query"],

        "query_protein": query_protein,

        "go_terms": [
            {
                "id": go["id"],
                "name": go["name"],
                "category": go["category"],
                "protein_accessions": [
                    p["accession"]
                    for p in go["proteins"][:10]
                ]
            }
            for go in go_terms[:30]
        ],

        "proteins": [
            {
                "accession": protein["accession"],
                "taxon_id": protein["taxon_id"],
                "name": protein["name"],
                "description": protein["description"],
                "gene": protein["gene"],
                "go_terms": protein["go_terms"],
                "entities": protein["entities"]
            }
            for protein in proteins[:100]
        ],

        "entities": [
            {
                "accession": entity["accession"],
                "name": entity["name"],
                "type": entity["type"],
                "description": entity["description"],
                "literatures": entity["literatures"][:10],
                "protein_accessions": [
                    p["accession"]
                    for p in entity["proteins"][:10]
                ]
            }
            for entity in entities[:100]
        ],

        "result_counts": {
            "proteins": len(proteins),
            "go_terms": len(go_terms),
            "entities": len(entities)
        },

        "context_truncated": {
            "proteins": len(proteins) > 100,
            "go_terms": len(go_terms) > 30,
            "entities": len(entities) > 100
        }
    }

    return json.dumps(
        context,
        default=str,
        ensure_ascii=False
    )

def agent_node(state: GraphState):
    state_context = build_state_context(state)
    system_message = SystemMessage(
        content=(
            SYSTEM_PROMPT
            + "\n\nCURRENT GRAPH STATE:\n"
            + state_context
        )
    )

    response = llm_with_tools.invoke([system_message] + state["messages"])
    if len(response.tool_calls) > 1:
        raise RuntimeError(
            "Model returned multiple parallel tool calls. "
            "Configure the endpoint for sequential tool calling."
        )

    return {
        "messages": [response]
    }


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent") 
    builder.add_conditional_edges("agent", tools_condition,{"tools": "tools", "__end__": END})
    builder.add_edge("tools", "agent")
    return builder.compile()

graph = build_graph()