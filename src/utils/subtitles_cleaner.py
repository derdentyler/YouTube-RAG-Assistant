import re
from typing import List, Dict, Union


def clean_subtitles(subtitles: List[Dict[str, Union[str, float]]]) -> List[Dict[str, Union[str, float]]]:
    """
    Очищает текст субтитров: убирает лишние пробелы, специальные символы и корректирует форматирование.

    :param subtitles: Список словарей с ключами 'text', 'start' и 'duration'.
    :return: Очищенный список субтитров.
    """
    cleaned_subtitles = []

    for entry in subtitles:
        cleaned_text = re.sub(r"\s+", " ", entry["text"]).strip()  # Убираем лишние пробелы
        cleaned_text = re.sub(r"\[.*?]|\(.*?\)", "", cleaned_text)  # Убираем текст в скобках (например, [музыка])

        if cleaned_text:  # Пропускаем пустые строки
            cleaned_subtitles.append({
                "text": cleaned_text,
                "start": entry["start"],
                "duration": entry["duration"]
            })

    return cleaned_subtitles
