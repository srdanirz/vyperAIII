import logging
from typing import Dict, Any, Optional, Tuple
import sqlite3
import psycopg2
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class DBAgent(BaseAgent):
    """
    Agente para realizar operaciones en bases de datos (SQLite, Postgres).
    """
    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.connections = {}
        self.current_db = None

    async def _execute(self) -> Dict[str, Any]:
        """Ejecuta la operación determinada: SELECT, INSERT, UPDATE, DELETE, etc."""
        try:
            operation_type = self._determine_operation()
            if operation_type == "query":
                return await self._execute_query()
            elif operation_type == "insert":
                return await self._execute_insert()
            elif operation_type == "update":
                return await self._execute_update()
            elif operation_type == "delete":
                return await self._execute_delete()
            else:
                return {
                    "error": f"Unsupported operation: {operation_type}",
                    "supported_operations": ["query", "insert", "update", "delete"]
                }
        except Exception as e:
            logger.error(f"Error in DBAgent: {e}", exc_info=True)
            raise

    def _determine_operation(self) -> str:
        """Determina el tipo de operación de base de datos necesaria"""
        task_lower = self.task.lower()
        
        if any(word in task_lower for word in ["select", "find", "search", "get"]):
            return "query"
        elif any(word in task_lower for word in ["insert", "add", "create"]):
            return "insert"
        elif any(word in task_lower for word in ["update", "modify", "change"]):
            return "update"
        elif any(word in task_lower for word in ["delete", "remove"]):
            return "delete"
        
        return "unknown"

    async def connect(self, connection_info: Dict[str, Any]) -> None:
        """Establece conexión con la base de datos"""
        db_type = connection_info.get("type", "sqlite")
        
        if db_type == "sqlite":
            conn = sqlite3.connect(connection_info["database"])
        elif db_type == "postgresql":
            conn = psycopg2.connect(
                host=connection_info["host"],
                database=connection_info["database"],
                user=connection_info["user"],
                password=connection_info["password"]
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        self.connections[connection_info["database"]] = conn
        self.current_db = connection_info["database"]

    async def _execute_query(self) -> Dict[str, Any]:
        """Ejecuta una consulta SELECT"""
        if not self.current_db:
            raise ValueError("No database connection established")
            
        conn = self.connections[self.current_db]
        cursor = conn.cursor()
        
        try:
            query = self._extract_query()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            return {
                "columns": columns,
                "data": data,
                "row_count": len(data)
            }
            
        finally:
            cursor.close()

    async def _execute_insert(self) -> Dict[str, Any]:
        """Ejecuta una operación INSERT"""
        if not self.current_db:
            raise ValueError("No database connection established")
            
        conn = self.connections[self.current_db]
        cursor = conn.cursor()
        
        try:
            query, data = self._extract_insert_data()
            cursor.execute(query, data)
            conn.commit()
            
            return {
                "inserted_id": cursor.lastrowid,
                "affected_rows": cursor.rowcount
            }
            
        finally:
            cursor.close()

    def _extract_query(self) -> str:
        """Extrae la consulta SQL del task o partial_data"""
        if "query" in self.partial_data:
            return self.partial_data["query"]
            
        # Aquí podrías implementar lógica para construir la consulta basada en el task
        return ""

    def _extract_insert_data(self) -> Tuple[str, List[Any]]:
        """Extrae la consulta INSERT y sus datos"""
        if "insert_data" in self.partial_data:
            data = self.partial_data["insert_data"]
            if isinstance(data, dict):
                columns = ", ".join(data.keys())
                placeholders = ", ".join(["?" for _ in data])
                query = f"INSERT INTO {data.get('table', '')} ({columns}) VALUES ({placeholders})"
                return query, list(data.values())
                
        return "", []

    async def disconnect(self) -> None:
        """Cierra las conexiones de base de datos"""
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()
        self.current_db = None
