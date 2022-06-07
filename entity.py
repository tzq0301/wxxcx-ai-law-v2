from typing import Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel


class SearchSamplesRequestBody(BaseModel):
    text: str
    accusations: List[str]


class TermOfImprisonment:
    def __init__(self,
                 imprisonment: int,
                 death_penalty: bool,
                 life_imprisonment: bool):
        self.imprisonment: int = imprisonment
        self.death_penalty: bool = death_penalty
        self.life_imprisonment: bool = life_imprisonment


class Meta:
    def __init__(self,
                 accusation: List[str],
                 term_of_imprisonment: Union[TermOfImprisonment, Dict],
                 relevant_articles: List[str],
                 criminals: List[str],
                 punish_of_money: int):
        self.accusation: List[str] = accusation
        self.term_of_imprisonment: TermOfImprisonment = TermOfImprisonment(**term_of_imprisonment)
        self.relevant_articles: List[str] = relevant_articles
        self.criminals: List[str] = criminals
        self.punish_of_money: int = punish_of_money


class Sample:
    def __init__(self,
                 meta: Union[Meta, Dict],
                 fact: str,
                 embedding: Optional[np.ndarray] = None):
        self.meta: Meta = Meta(**meta)
        self.fact: str = fact
        self.embedding: Optional[np.ndarray] = embedding
