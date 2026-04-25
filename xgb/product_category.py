"""
Blood product name → supply category classifier.
Uses priority-ordered keyword matching on blood_product_name.
"""

CATEGORY_RULES = [
    ("红细胞类", ["红细胞", "悬浮红", "浓缩红", "洗涤红", "去白红", "RBC"]),
    ("血小板类", ["血小板", "单采", "机采血小板", "浓缩血小板", "PLT"]),
    ("血浆类", ["血浆", "冷沉淀", "FFP", "病毒灭活"]),
]

ALL_CATEGORIES = [r[0] for r in CATEGORY_RULES]


def classify_product(name: str) -> str:
    """Classify a blood_product_name into a supply category.

    Priority-ordered: first match wins.
    Returns '其他' if no rule matches.
    """
    if not name:
        return "其他"
    upper = name.strip().upper()
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw.upper() in upper:
                return category
    return "其他"
