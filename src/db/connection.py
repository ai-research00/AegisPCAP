"""
AegisPCAP database initialization and connection pooling
"""
import logging
from typing import Optional
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection management with pooling"""
    
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 40):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine = None
        self.SessionLocal = None
    
    def initialize(self):
        """Initialize database connection with optimal settings"""
        try:
            # Determine pool class based on database type
            if 'sqlite' in self.database_url:
                # SQLite uses static pool for in-memory or file-based
                pool_class = StaticPool
                pool_size = 1
                max_overflow = 0
            else:
                # PostgreSQL, MySQL use queue pool
                pool_class = QueuePool
                pool_size = self.pool_size
                max_overflow = self.max_overflow
            
            self.engine = create_engine(
                self.database_url,
                poolclass=pool_class,
                pool_size=pool_size,
                max_overflow=max_overflow,
                echo=False,
                pool_pre_ping=True,  # Verify connections before using
                connect_args={
                    'connect_timeout': 10,
                    'options': '-c default_transaction_isolation=read_committed',
                } if 'postgresql' in self.database_url else {}
            )
            
            # Event listeners for connection management
            @event.listens_for(self.engine, "connect")
            def receive_connect(dbapi_conn, connection_record):
                """Configure connection on creation"""
                if 'postgresql' in self.database_url:
                    # Set isolation level
                    dbapi_conn.isolation_level = 'READ_COMMITTED'
            
            @event.listens_for(self.engine, "pool_connect")
            def receive_pool_connect(dbapi_conn, connection_record):
                """Configure pooled connection"""
                if 'postgresql' in self.database_url:
                    cursor = dbapi_conn.cursor()
                    cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED;")
                    cursor.close()
            
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info(f"Database initialized: {self.database_url}")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def get_session(self) -> Session:
        """Get new database session"""
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.SessionLocal()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_pool_status(self) -> dict:
        """Get connection pool status"""
        if self.engine is None:
            return {}
        
        try:
            pool = self.engine.pool
            return {
                'pool_size': pool.size() if hasattr(pool, 'size') else 'N/A',
                'checkedout': pool.checkedout() if hasattr(pool, 'checkedout') else 'N/A',
                'overflow': pool.overflow() if hasattr(pool, 'overflow') else 'N/A',
                'checked_in': pool.checked_in() if hasattr(pool, 'checked_in') else 'N/A',
            }
        except Exception as e:
            logger.error(f"Pool status check failed: {e}")
            return {}
    
    def dispose(self):
        """Close all connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections disposed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()


class ContextManager:
    """Database context manager for safe session handling"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.session = None
    
    def __enter__(self) -> Session:
        self.session = self.db.get_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            self.session.close()


# Dependency injection helper
class DatabaseProvider:
    """Provides database instances across application"""
    
    _instance: Optional[DatabaseConnection] = None
    
    @classmethod
    def initialize(cls, database_url: str) -> DatabaseConnection:
        """Initialize singleton database connection"""
        if cls._instance is None:
            cls._instance = DatabaseConnection(database_url)
            cls._instance.initialize()
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> DatabaseConnection:
        """Get database instance"""
        if cls._instance is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton (for testing)"""
        if cls._instance:
            cls._instance.dispose()
        cls._instance = None


# Middleware for request lifecycle management
class DatabaseMiddleware:
    """ASGI middleware for automatic session management"""
    
    def __init__(self, app, db_connection: DatabaseConnection):
        self.app = app
        self.db = db_connection
    
    async def __call__(self, scope, receive, send):
        """Handle request with database session"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Create session for this request
        session = self.db.get_session()
        scope["db_session"] = session
        
        async def send_with_cleanup(message):
            """Send response and cleanup session"""
            await send(message)
            if message.get("type") == "http.response.body":
                session.close()
        
        await self.app(scope, receive, send_with_cleanup)


# Query optimization helpers
class QueryOptimizer:
    """Helper class for optimized database queries"""
    
    @staticmethod
    def batch_insert(session: Session, model_class, data_list: list, batch_size: int = 1000):
        """Insert large amount of data efficiently"""
        try:
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                session.bulk_insert_mappings(model_class, batch)
                session.commit()
            logger.info(f"Batch inserted {len(data_list)} records")
            return len(data_list)
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            session.rollback()
            return 0
    
    @staticmethod
    def bulk_update(session: Session, model_class, updates: list, batch_size: int = 1000):
        """Update large amount of data efficiently"""
        try:
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i + batch_size]
                session.bulk_update_mappings(model_class, batch)
                session.commit()
            logger.info(f"Batch updated {len(updates)} records")
            return len(updates)
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            session.rollback()
            return 0
    
    @staticmethod
    def paginate(query, page: int = 1, page_size: int = 50):
        """Paginate query results"""
        total = query.count()
        items = query.offset((page - 1) * page_size).limit(page_size).all()
        
        return {
            'items': items,
            'total': total,
            'page': page,
            'page_size': page_size,
            'pages': (total + page_size - 1) // page_size,
        }


# Exception handling
class DatabaseException(Exception):
    """Base database exception"""
    pass


class ConnectionException(DatabaseException):
    """Connection error"""
    pass


class QueryException(DatabaseException):
    """Query execution error"""
    pass
