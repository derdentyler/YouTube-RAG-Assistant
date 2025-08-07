from typing import Protocol, List, Union


class Embedder(Protocol):
    """
    Protocol (интерфейс) для любых моделей эмбеддинга.
    Определяет, что у объекта должен быть метод `.encode()`.
    """

    def encode(
            self,
            texts: Union[str, List[str]],
            *,
            convert_to_tensor: bool = False
    ) -> Union[List[float], List[List[float]]]:
        """
        Преобразовать текст или список текстов в вектор(ы).

        :param texts: либо одиночная строка, либо список строк
        :param convert_to_tensor: должны ли выходные векторы быть тензорами
        :return: список чисел (вектор) или список списков (для нескольких текстов)
        """
        pass
