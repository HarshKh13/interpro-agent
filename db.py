from sqlalchemy import Column, Table, create_engine, ForeignKey, String, Integer, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
import json
from typing import Optional

engine = create_engine(
    "postgresql+psycopg://interpro_user:interpro@localhost:5432/interpro"
)

class Base(DeclarativeBase):
    pass

class GoTerms(Base):
    __tablename__ = "go_terms"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    entities: Mapped[list["Entity"]] = relationship(secondary="entity_to_go_term_map",back_populates="go_terms")
    proteins: Mapped[list["Protein"]] = relationship(secondary="protein_to_go_term_map",back_populates="go_terms")
    
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

# with engine.connect() as conn:
#     conn.execute(text("DROP TABLE IF EXISTS entity CASCADE"))
#     conn.execute(text("DROP TABLE IF EXISTS go_terms CASCADE"))
#     conn.execute(text("DROP TABLE IF EXISTS literature CASCADE"))
#     conn.execute(text("DROP TABLE IF EXISTS organism CASCADE"))
#     conn.execute(text("DROP TABLE IF EXISTS protein CASCADE"))
#     conn.execute(text("DROP TABLE IF EXISTS entity_to_go_term_map CASCADE"))
#     conn.execute(text("DROP TABLE IF EXISTS protein_to_go_term_map CASCADE"))
#     conn.execute(text("DROP TABLE IF EXISTS protein_to_entity_map CASCADE"))
#     conn.execute(text("DROP TABLE IF EXISTS entity_to_literature_map CASCADE"))
#     conn.commit()

# Base.metadata.create_all(engine)

# with open('interpro_fulldata.json', 'r', encoding='utf-8') as file:
#     entity_data = json.load(file)

# with open('arabidopsis_protein_interpro_relations.json', 'r', encoding='utf-8') as file:
#     protein_data = json.load(file)

# with Session(engine) as session:
#     for index,entity_record in enumerate(entity_data):
#         entity = Entity(
#             accession=entity_record.get("accession"),
#             type=entity_record.get("type"),
#             name=entity_record.get("name").get("name"),
#             description=(entity_record.get("description")[0].get("text") if len(entity_record.get("description"))>0 else None)
#         )
#         if entity_record.get("go_terms") is not None:
#             for go_term in entity_record.get("go_terms"):
#                 existing_go_term = session.get(GoTerms,go_term.get("identifier"))
#                 if existing_go_term:
#                     entity.go_terms.append(existing_go_term)
#                 else:
#                     entity.go_terms.append(
#                         GoTerms(
#                             id=go_term.get("identifier"),
#                             name=go_term.get("name"),
#                             category=go_term.get("category").get("name")
#                         )
#                     )
        
#         if entity_record.get("literature") is not None:
#             for literature_id,literature in entity_record.get("literature").items():
#                 existing_literature = session.get(Literature,literature_id)
#                 if existing_literature:
#                     entity.literatures.append(existing_literature)
#                 else:
#                     entity.literatures.append(
#                         Literature(
#                             id=literature_id,
#                             title=literature.get("title"),
#                             url=literature.get("DOI_URL") if literature.get("DOI_URL") is not None else None
#                         )
#                     ) 
    
#         session.add(entity)
#         session.commit()
#         print(f"{index}/{len(entity_data)}",end="\r")

# with Session(engine) as session:
#     for index,protein_record in enumerate(protein_data):
#         protein = Protein(
#             accession=protein_record.get("accession"),
#             name=protein_record.get("name"),
#             description=protein_record.get("description"),
#             length=protein_record.get("length"),
#             sequence=protein_record.get("sequence"),
#             proteome=protein_record.get("proteome"),
#             gene=protein_record.get("gene")
#         )

#         existing_organism = session.get(Organism,protein_record.get("source_organism").get("taxId"))
#         if existing_organism:
#             protein.organism = existing_organism
#         else:
#             protein.organism = Organism(
#                 taxon_id=protein_record.get("source_organism").get("taxId"),
#                 scientific_name=protein_record.get("source_organism").get("scientificName"),
#                 full_name=protein_record.get("source_organism").get("fullName")
#             )

#         if protein_record.get("go_terms") is not None:
#             for go_term in protein_record.get("go_terms"):
#                 existing_go_term = session.get(GoTerms,go_term.get("identifier"))
#                 if existing_go_term:
#                     protein.go_terms.append(existing_go_term)
#                 else:
#                     protein.go_terms.append(
#                         GoTerms(
#                             id=go_term.get("identifier"),
#                             name=go_term.get("name"),
#                             category=go_term.get("category").get("name")
#                         )
#                     )
        

#         if protein_record.get("dbRelations") is not None and protein_record.get("dbRelations").get("interpro") is not None:
#             for entity_id in protein_record.get("dbRelations").get("interpro"):
#                 entity = session.get(Entity,entity_id)
#                 protein.entities.append(entity)

#         session.add(protein)
#         session.commit()
#         print(f"{index}/{len(protein_data)}",end="\r")



with Session(engine) as session:
    proteins = session.query(Protein).filter(Protein.taxon_id == "3702").all()
    with open("arabidopsis_proteins.fasta", "w") as f:
        for protein in proteins:
            f.write(f">{protein.accession}\n")
            f.write(f"{protein.sequence}\n")