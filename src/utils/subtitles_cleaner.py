import re
from typing import List, Dict, Union

def clean_subtitles(
    subtitles: List[Dict[str, Union[str, float]]]
) -> List[Dict[str, Union[str, float]]]:
    """
    Очищает текст субтитров: убирает HTML-теги, скобки, лишние пробелы.
    """
    cleaned = []
    for entry in subtitles:
        txt = entry["text"]

        # Удаляем HTML-теги (например, <c>, <00:00:01.140> и т.д.)
        txt = re.sub(r"<[^>]+>", "", txt)

        # Удаляем текст в скобках
        txt = re.sub(r"\[.*?]|\(.*?\)", "", txt)

        # Удаляем таймкоды типа <00:02:44.340>
        txt = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", txt)

        # Сводим пробелы
        txt = re.sub(r"\s+", " ", txt).strip()

        if txt:
            cleaned.append({
                "text": txt,
                "start": entry["start"],
                "duration": entry["duration"]
            })

    return cleaned
