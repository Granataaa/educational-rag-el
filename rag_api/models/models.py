from pydantic import BaseModel

class Entity(BaseModel):
    text: str
    type: str
    start_char: int
    end_char: int

class linked_entity(BaseModel):
    mention_text: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    ner_type: str | None = None
    wikidata_id: str | None = None
    wikidata_label: str | None = None
    wikidata_description: str | None = None
    final_score: float | None = None
    all_candidates: list[dict] = []

class Chunk(BaseModel):
    chunk_id: int
    text: str
    source: str
    start_time: float | None = None
    end_time: float | None = None
    entities: list[Entity] = []
    linked_entities: list[linked_entity] = []

class RagResponse(BaseModel):
    testoRisp: str | None = None
    chunks: list[Chunk] = []