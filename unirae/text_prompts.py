from typing import Iterable, List

DEFAULT_TEMPLATES = [
    "a photo of a {class}",
]


def normalize_class_name(name: str) -> str:
    return name.replace("_", " ").strip()


def build_prompts(class_names: Iterable[str], templates: List[str] = None) -> List[List[str]]:
    templates = templates or DEFAULT_TEMPLATES
    all_prompts = []
    for cls in class_names:
        cname = normalize_class_name(cls)
        all_prompts.append([tpl.format(**{"class": cname}) for tpl in templates])
    return all_prompts
