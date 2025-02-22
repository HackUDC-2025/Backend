def get_technical_description(level: str) -> str:
    return {
        "bajo": "explicaciones simples sin jerga especializada",
        "medio": "conceptos fundamentales con algunos términos técnicos",
        "alto": "análisis profundo con vocabulario especializado",
    }[level]


def get_language_instruction(style: str) -> str:
    return {
        "sencillo": "tono coloquial y cercano, usando analogías cotidianas",
        "formal": "lenguaje culto pero accesible, manteniendo profesionalidad",
        "detallado": "terminología precisa con referencias académicas",
    }[style]
