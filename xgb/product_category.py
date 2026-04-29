"""
Blood product name → supply category classifier.
Uses priority-ordered keyword matching on blood_product_name.
"""

CATEGORY_RULES = [
    ("红细胞类", ["洗涤红细胞", "悬浮红细胞", "悬浮少白细胞红细胞", "辐照悬浮少白细胞红细胞", "辐照洗涤红细胞", "辐照悬浮红细胞","冰冻解冻去甘油红细胞","浓缩白细胞","辐照去甘油解冻红细胞"]),
    ("血小板类", ["单采血小板", "辐照单采血小板", "浓缩血小板", "辐照浓缩血小板", "去白细胞单采血小板","辐照去白细胞单采血小板"]),
    ("血浆类", ["新鲜冰冻血浆", "普通血浆", "单采新鲜冰冻血浆", "去白新鲜冰冻血浆","病毒灭活新冠恢复期冰冻血浆","单采新冠康复者血浆","冰冻新冠康复者血浆","病毒灭活冰冻血浆"]),
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
