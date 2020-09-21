from enum import Enum, unique
@unique
class VectEnum(Enum):
    TfidfVectorizer = 1
    CountVectorizer = 2