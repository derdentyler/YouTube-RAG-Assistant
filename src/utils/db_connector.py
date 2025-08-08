import os
from typing import List, Tuple, Optional
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extensions import connection as PGConnection
from dotenv import load_dotenv
from src.utils.logger_loader import LoggerLoader


load_dotenv()
logger = LoggerLoader.get_logger()

USER = os.getenv("USER")  # Если USER нет в .env, используем "default_user"
PASSWORD = os.getenv("SUPABASE_KEY")
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))  # Если PORT нет в .env, используем 5432
DBNAME = os.getenv("DBNAME")


print(f"USER: {USER}")
print(f"HOST: {HOST}")
print(f"PORT: {PORT}")
print(f"DBNAME: {DBNAME}")


class DBConnector:
    def __init__(self) -> None:
        self._pool: Optional[SimpleConnectionPool] = None

        try:
            logger.info("Инициализация пула соединений...")

            self._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME,
            )

            if self._pool:
                logger.info("Пул соединений успешно создан.")
                self.initialize_db()
            else:
                raise Exception("Не удалось создать пул соединений.")

        except Exception as error:
            logger.error(f"Ошибка при инициализации пула соединений: {error}")
            raise

    def get_connection(self) -> PGConnection:
        """Получить соединение из пула."""
        try:
            if not self._pool:
                raise RuntimeError("Пул соединений не инициализирован.")
            conn = self._pool.getconn()
            logger.info("Соединение получено из пула.")
            return conn
        except Exception as error:
            logger.error(f"Ошибка при получении соединения: {error}")
            raise

    def release_connection(self, connection: PGConnection) -> None:
        """Вернуть соединение обратно в пул."""
        try:
            if self._pool and connection:
                self._pool.putconn(connection)
                logger.info("Соединение возвращено в пул.")
        except Exception as error:
            logger.error(f"Ошибка при возврате соединения: {error}")

    def close(self) -> None:
        """Закрыть все соединения пула."""
        try:
            if self._pool:
                self._pool.closeall()
                logger.info("Пул соединений закрыт.")
        except Exception as error:
            logger.error(f"Ошибка при закрытии пула соединений: {error}")

    def initialize_db(self) -> None:
        """Создание таблицы и расширения, если нужно."""
        conn = None
        try:
            logger.info("Инициализация базы данных...")
            self.ensure_pgvector_extension()
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT to_regclass('public.subtitles');")
                exists = cursor.fetchone()

                if not exists[0]:
                    logger.warning("Таблица 'subtitles' не найдена. Создаём...")
                    self.create_subtitles_table(conn)
                else:
                    logger.info("Таблица 'subtitles' уже существует.")

        except Exception as error:
            logger.error(f"Ошибка при инициализации БД: {error}")
            raise

        finally:
            if conn:
                self.release_connection(conn)

    def ensure_pgvector_extension(self) -> None:
        """Установить pgvector, если он ещё не установлен."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                if not cursor.fetchone():
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    conn.commit()
                    logger.info("Расширение 'pgvector' установлено.")
                else:
                    logger.info("Расширение 'pgvector' уже установлено.")

        except Exception as error:
            logger.error(f"Ошибка при установке pgvector: {error}")

        finally:
            if conn:
                self.release_connection(conn)

    def create_subtitles_table(self, connection: PGConnection) -> None:
        """Создать таблицу 'subtitles'."""
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS subtitles (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        video_id TEXT NOT NULL,
                        start_time FLOAT NOT NULL,
                        end_time FLOAT NOT NULL,
                        text TEXT NOT NULL,
                        embedding VECTOR(768)
                    );
                """)
                connection.commit()
            logger.info("Таблица 'subtitles' успешно создана.")
        except Exception as error:
            logger.error(f"Ошибка при создании таблицы: {error}")
            connection.rollback()

    def insert_subtitle(
            self, video_id: str, start_time: float, end_time: float,
            text: str, embedding: List[float]
    ) -> None:
        """Добавить субтитры в таблицу."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO subtitles (video_id, start_time, end_time, text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                """, (video_id, start_time, end_time, text, embedding))
                conn.commit()

            logger.info(f"Субтитры для {video_id} успешно добавлены.")

        except Exception as error:
            logger.error(f"Ошибка при вставке субтитров: {error}")

        finally:
            if conn:
                self.release_connection(conn)

    def search_similar_embeddings(self, embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Поиск похожих субтитров по embedding."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT text, 1 - (embedding <#> %s) AS similarity
                    FROM subtitles
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (embedding, top_k))

                return cursor.fetchall()

        except Exception as error:
            logger.error(f"Ошибка при поиске эмбеддингов: {error}")
            return []

        finally:
            if conn:
                self.release_connection(conn)

    def drop_table(self) -> None:
        """Удалить таблицу 'subtitles'."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS subtitles;")
                conn.commit()
            logger.info("Таблица 'subtitles' удалена.")

        except Exception as error:
            logger.error(f"Ошибка при удалении таблицы: {error}")

        finally:
            if conn:
                self.release_connection(conn)

    def fetch_subtitles(self, video_id: str) -> List[Tuple[str]]:
        """Получить субтитры по video_id."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM subtitles WHERE video_id = %s LIMIT 1;", (video_id,))
                if not cursor.fetchone():
                    logger.warning(f"Субтитры для {video_id} не найдены.")
                    return []

                cursor.execute("SELECT text FROM subtitles WHERE video_id = %s;", (video_id,))
                subtitles = cursor.fetchall()
            return subtitles

        except Exception as error:
            logger.error(f"Ошибка при извлечении субтитров: {error}")
            return []

        finally:
            if conn:
                self.release_connection(conn)

    def clear_table(self) -> None:
        """Удалить все записи из таблицы."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM subtitles;")
                conn.commit()
            logger.info("Таблица 'subtitles' очищена.")

        except Exception as error:
            logger.error(f"Ошибка при очистке таблицы: {error}")

        finally:
            if conn:
                self.release_connection(conn)
