import re
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from src.core.abstractions.embeddings import Embedder
from src.utils.logger_loader import LoggerLoader


class SemanticChunker:
    """
    Semantic chunking для субтитров видео.
    
    Разбивает текст на чанки на основе семантической близости,
    сохраняя целостность тем и контролируя размер чанков.
    """
    
    def __init__(
        self,
        embedding_model: Embedder,
        max_tokens: int = 150,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 50
    ):
        """
        Инициализация SemanticChunker.
        
        Args:
            embedding_model: Модель для вычисления эмбеддингов
            max_tokens: Максимальный размер чанка в токенах (примерно)
            similarity_threshold: Порог семантической близости для объединения предложений
            min_chunk_size: Минимальный размер чанка в токенах
        """
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.logger = LoggerLoader.get_logger()
        
    def _estimate_tokens(self, text: str) -> int:
        """
        Оценка количества токенов в тексте.
        Использует приближение: ~4 символа на токен для большинства языков.
        """
        # Простое приближение: делим длину на 4
        # Можно улучшить, используя tiktoken или другие библиотеки
        return len(text) // 4
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения.
        Использует простой regex для русского и английского языков.
        """
        # Паттерн для разбиения на предложения
        # Учитывает точки, восклицательные и вопросительные знаки
        # Исключает сокращения типа "т.д.", "т.п." и т.д.
        pattern = r'(?<=[.!?])\s+(?=[А-ЯA-Z])'
        sentences = re.split(pattern, text)
        
        # Фильтруем пустые предложения и очищаем
        cleaned = [s.strip() for s in sentences if s.strip()]
        return cleaned
    
    def _compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Вычисление эмбеддингов для списка предложений.
        
        Args:
            sentences: Список предложений
            
        Returns:
            numpy array с эмбеддингами формы (n_sentences, embedding_dim)
        """
        if not sentences:
            return np.array([])
        
        # Вычисляем эмбеддинги батчами для эффективности
        embeddings = self.embedding_model.encode(sentences, convert_to_tensor=False)
        
        # Преобразуем в numpy array если нужно
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu().numpy()
        elif not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Если один текст - делаем 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        return embeddings
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычисление косинусного сходства между двумя векторами.
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def chunk(
        self, 
        segments: List[Dict[str, Union[str, float]]]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Основной метод семантического чанкинга.
        
        Args:
            segments: Список сегментов субтитров с полями:
                - text: текст сегмента
                - start: время начала
                - duration: длительность
                
        Returns:
            Список чанков с полями:
                - text: текст чанка
                - start: время начала первого сегмента
                - duration: общая длительность чанка
        """
        if not segments:
            return []
        
        # 1. Объединяем все сегменты в один текст с сохранением временных меток
        full_text_parts = []
        segment_info = []  # Информация о каждом сегменте для маппинга
        
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
                
            start_time = seg.get("start", 0.0)
            duration = seg.get("duration", 0.0)
            end_time = start_time + duration
            
            full_text_parts.append(text)
            segment_info.append({
                "text": text,
                "start_time": start_time,
                "end_time": end_time
            })
        
        full_text = " ".join(full_text_parts)
        
        if not full_text.strip():
            return []
        
        # 2. Разбиваем на предложения
        sentences = self._split_into_sentences(full_text)
        if not sentences:
            # Если не удалось разбить на предложения, используем весь текст
            sentences = [full_text]
        
        self.logger.info(f"Split into {len(sentences)} sentences")
        
        # 3. Вычисляем эмбеддинги для всех предложений
        sentence_embeddings = self._compute_sentence_embeddings(sentences)
        
        if len(sentence_embeddings) == 0:
            return []
        
        # 4. Создаем маппинг предложений на сегменты для вычисления временных меток
        # Используем позиции в полном тексте для более точного маппинга
        sentence_to_segments = []
        current_pos = 0
        
        for sentence in sentences:
            # Находим позицию предложения в полном тексте
            sentence_start = full_text.find(sentence, current_pos)
            if sentence_start == -1:
                # Если не найдено, используем все сегменты
                sentence_to_segments.append(segment_info)
                continue
            
            sentence_end = sentence_start + len(sentence)
            current_pos = sentence_end
            
            # Находим сегменты, которые пересекаются с этим предложением
            matching_segments = []
            seg_pos = 0
            for seg_info in segment_info:
                seg_text = seg_info["text"]
                seg_start = full_text.find(seg_text, seg_pos)
                if seg_start == -1:
                    continue
                seg_end = seg_start + len(seg_text)
                seg_pos = seg_end
                
                # Проверяем пересечение
                if not (sentence_end <= seg_start or sentence_start >= seg_end):
                    matching_segments.append(seg_info)
            
            # Если не нашли совпадений, используем ближайший сегмент
            if not matching_segments and segment_info:
                # Находим сегмент, который ближе всего к позиции предложения
                best_seg = None
                min_dist = float('inf')
                seg_pos = 0
                for seg_info in segment_info:
                    seg_text = seg_info["text"]
                    seg_start = full_text.find(seg_text, seg_pos)
                    if seg_start == -1:
                        continue
                    seg_end = seg_start + len(seg_text)
                    seg_pos = seg_end
                    
                    # Расстояние до центра сегмента
                    seg_center = (seg_start + seg_end) / 2
                    sent_center = (sentence_start + sentence_end) / 2
                    dist = abs(seg_center - sent_center)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_seg = seg_info
                
                if best_seg:
                    matching_segments.append(best_seg)
            
            sentence_to_segments.append(matching_segments if matching_segments else segment_info)
        
        # 5. Создаем чанки на основе семантической близости
        chunks = []
        current_chunk_sentences = []
        current_chunk_embeddings = []
        current_chunk_tokens = 0
        current_chunk_sentence_indices = []  # Индексы предложений в текущем чанке
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            sentence_emb = sentence_embeddings[i]
            
            # Проверяем, нужно ли начать новый чанк
            should_start_new = False
            
            if current_chunk_sentences:
                # Вычисляем similarity с последним предложением в текущем чанке
                last_emb = current_chunk_embeddings[-1]
                similarity = self._cosine_similarity(sentence_emb, last_emb)
                
                # Если similarity низкая или чанк слишком большой - начинаем новый
                if similarity < self.similarity_threshold:
                    should_start_new = True
                    self.logger.debug(
                        f"Low similarity ({similarity:.3f}) at sentence {i}, starting new chunk"
                    )
                elif current_chunk_tokens + sentence_tokens > self.max_tokens:
                    should_start_new = True
                    self.logger.debug(
                        f"Max tokens reached ({current_chunk_tokens + sentence_tokens}), starting new chunk"
                    )
            else:
                # Первое предложение - всегда начинаем новый чанк
                should_start_new = False
            
            if should_start_new and current_chunk_sentences:
                # Сохраняем текущий чанк, если он достаточно большой
                if current_chunk_tokens >= self.min_chunk_size:
                    chunk_text = " ".join(current_chunk_sentences)
                    # Вычисляем временные метки для чанка на основе индексов предложений
                    chunk_start, chunk_end = self._get_chunk_time_from_sentences(
                        current_chunk_sentence_indices, sentence_to_segments
                    )
                    chunks.append({
                        "text": chunk_text,
                        "start": chunk_start,
                        "duration": chunk_end - chunk_start
                    })
                else:
                    # Если чанк слишком маленький, все равно сохраняем его
                    # (можно объединить с предыдущим, но для простоты сохраняем отдельно)
                    chunk_text = " ".join(current_chunk_sentences)
                    chunk_start, chunk_end = self._get_chunk_time_from_sentences(
                        current_chunk_sentence_indices, sentence_to_segments
                    )
                    chunks.append({
                        "text": chunk_text,
                        "start": chunk_start,
                        "duration": chunk_end - chunk_start
                    })
                
                # Начинаем новый чанк
                current_chunk_sentences = [sentence]
                current_chunk_embeddings = [sentence_emb]
                current_chunk_tokens = sentence_tokens
                current_chunk_sentence_indices = [i]
            else:
                # Добавляем предложение в текущий чанк
                current_chunk_sentences.append(sentence)
                current_chunk_embeddings.append(sentence_emb)
                current_chunk_tokens += sentence_tokens
                current_chunk_sentence_indices.append(i)
        
        # Сохраняем последний чанк
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_start, chunk_end = self._get_chunk_time_from_sentences(
                current_chunk_sentence_indices, sentence_to_segments
            )
            chunks.append({
                "text": chunk_text,
                "start": chunk_start,
                "duration": chunk_end - chunk_start
            })
        
        self.logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _get_chunk_time_from_sentences(
        self,
        sentence_indices: List[int],
        sentence_to_segments: List[List[Dict[str, float]]]
    ) -> Tuple[float, float]:
        """
        Вычисляет временные границы чанка на основе индексов предложений.
        
        Args:
            sentence_indices: Индексы предложений, входящих в чанк
            sentence_to_segments: Маппинг индексов предложений на сегменты
            
        Returns:
            Tuple (start_time, end_time)
        """
        if not sentence_indices:
            return 0.0, 0.0
        
        # Собираем все сегменты, связанные с предложениями в чанке
        all_segments = []
        for idx in sentence_indices:
            if idx < len(sentence_to_segments):
                all_segments.extend(sentence_to_segments[idx])
        
        if not all_segments:
            return 0.0, 0.0
        
        # Находим минимальное start_time и максимальное end_time
        start_time = min(seg["start_time"] for seg in all_segments)
        end_time = max(seg["end_time"] for seg in all_segments)
        
        return start_time, end_time

