import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import uuid
import redis
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import pinecone
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import schedule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QueryPattern:
    metric1: str
    metric2: str
    strength: float
    count: int
    last_seen: datetime

@dataclass
class QueryRecommendation:
    type: str  # 'related_metric', 'optimization', 'context'
    message: str
    confidence: float
    metadata: Dict[str, Any]

class QueryPatternLogger:
    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.session_queries = {}
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Create tables if they don't exist"""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS query_logs (
            id VARCHAR(36) PRIMARY KEY,
            session_id VARCHAR(255),
            user_id VARCHAR(255),
            timestamp TIMESTAMP,
            natural_language_query TEXT,
            generated_sql TEXT,
            sql_hash VARCHAR(64),
            execution_time_ms INTEGER,
            result_row_count INTEGER,
            query_category VARCHAR(100),
            business_entities TEXT,
            query_complexity_score INTEGER,
            success BOOLEAN,
            error_message TEXT
        );
        
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255),
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            total_queries INTEGER,
            session_type VARCHAR(50)
        );
        
        CREATE TABLE IF NOT EXISTS query_sequences (
            id VARCHAR(36) PRIMARY KEY,
            session_id VARCHAR(255),
            query_order INTEGER,
            query_id VARCHAR(36),
            time_gap_seconds INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS metric_relationships (
            id VARCHAR(36) PRIMARY KEY,
            metric1 VARCHAR(100),
            metric2 VARCHAR(100),
            strength FLOAT,
            co_occurrence_count INTEGER,
            last_updated TIMESTAMP,
            UNIQUE(metric1, metric2)
        );
        """
        
        with self.engine.connect() as conn:
            for statement in create_tables_sql.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()
    
    def log_query(self, session_id: str, user_id: str, nl_query: str, 
                  generated_sql: str, execution_time: Optional[int] = None,
                  result_count: Optional[int] = None, success: bool = True, 
                  error: Optional[str] = None) -> str:
        
        sql_hash = hashlib.sha256(self._normalize_sql(generated_sql).encode()).hexdigest()[:16]
        entities = self._extract_business_entities(nl_query)
        category = self._categorize_query(nl_query, entities)
        complexity = self._calculate_complexity(generated_sql)
        
        query_id = str(uuid.uuid4())
        
        insert_sql = """
        INSERT INTO query_logs VALUES (
            :id, :session_id, :user_id, :timestamp, :nl_query, :sql, :sql_hash,
            :exec_time, :result_count, :category, :entities, :complexity, :success, :error
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(insert_sql), {
                'id': query_id,
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': datetime.utcnow(),
                'nl_query': nl_query,
                'sql': generated_sql,
                'sql_hash': sql_hash,
                'exec_time': execution_time,
                'result_count': result_count,
                'category': category,
                'entities': json.dumps(entities),
                'complexity': complexity,
                'success': success,
                'error': error
            })
            conn.commit()
        
        return query_id
    
    def _normalize_sql(self, sql: str) -> str:
        import re
        normalized = re.sub(r"'[^']*'", "'VALUE'", sql)
        normalized = re.sub(r'\b\d+\b', 'NUMBER', normalized)
        normalized = re.sub(r'\s+', ' ', normalized.strip().upper())
        return normalized
    
    def _extract_business_entities(self, nl_query: str) -> Dict[str, List[str]]:
        business_keywords = {
            'metrics': ['sales', 'revenue', 'profit', 'cost', 'churn', 'conversion', 'margin'],
            'dimensions': ['region', 'product', 'customer', 'time', 'channel', 'segment'],
            'time_periods': ['monthly', 'quarterly', 'yearly', 'last month', 'YTD', 'week'],
            'operations': ['sum', 'average', 'count', 'trend', 'compare', 'growth']
        }
        
        entities = {}
        for category, keywords in business_keywords.items():
            found = [kw for kw in keywords if kw.lower() in nl_query.lower()]
            if found:
                entities[category] = found
        
        return entities
    
    def _categorize_query(self, nl_query: str, entities: Dict) -> str:
        if 'sales' in str(entities).lower() or 'revenue' in str(entities).lower():
            return 'sales'
        elif 'customer' in str(entities).lower() or 'churn' in str(entities).lower():
            return 'customer'
        elif 'marketing' in nl_query.lower() or 'conversion' in str(entities).lower():
            return 'marketing'
        elif 'finance' in nl_query.lower() or 'profit' in str(entities).lower():
            return 'finance'
        else:
            return 'general'
    
    def _calculate_complexity(self, sql: str) -> int:
        complexity = 1
        complexity += sql.upper().count('JOIN') * 2
        complexity += sql.upper().count('SUBQUERY') * 3
        complexity += sql.upper().count('GROUP BY')
        complexity += sql.upper().count('ORDER BY')
        complexity += sql.upper().count('HAVING')
        return min(complexity, 10)

class SessionManager:
    def __init__(self, db_engine, redis_client):
        self.db = db_engine
        self.redis = redis_client
        self.active_sessions = {}
    
    def start_session(self, user_id: str, session_type: str = 'ad-hoc') -> str:
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'start_time': datetime.utcnow().isoformat(),
            'session_type': session_type,
            'query_count': '0'
        }
        
        if self.redis:
            self.redis.hset(f"session:{session_id}", mapping=session_data)
            self.redis.expire(f"session:{session_id}", 3600)
        
        self.active_sessions[session_id] = {
            'queries': [],
            'start_time': datetime.utcnow()
        }
        
        logger.info(f"Started session {session_id} for user {user_id}")
        return session_id
    
    def add_query_to_session(self, session_id: str, query_id: str):
        if self.redis:
            self.redis.hincrby(f"session:{session_id}", "query_count", 1)
        
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['queries'].append({
                'query_id': query_id,
                'timestamp': datetime.utcnow()
            })
    
    def end_session(self, session_id: str):
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            
            # Store session summary in database
            with self.db.connect() as conn:
                conn.execute(text("""
                    INSERT INTO user_sessions VALUES 
                    (:session_id, :user_id, :start_time, :end_time, :total_queries, :session_type)
                    ON CONFLICT (session_id) DO UPDATE SET
                    end_time = :end_time, total_queries = :total_queries
                """), {
                    'session_id': session_id,
                    'user_id': self.redis.hget(f"session:{session_id}", 'user_id') if self.redis else 'unknown',
                    'start_time': session_data['start_time'],
                    'end_time': datetime.utcnow(),
                    'total_queries': len(session_data['queries']),
                    'session_type': 'ad-hoc'
                })
                conn.commit()
            
            del self.active_sessions[session_id]
            
            if self.redis:
                self.redis.delete(f"session:{session_id}")
            
            logger.info(f"Ended session {session_id}")

class BusinessMetricAnalyzer:
    def __init__(self, pinecone_index, db_engine, model_name: str = 'all-MiniLM-L6-v2'):
        self.index = pinecone_index
        self.db = db_engine
        self.encoder = SentenceTransformer(model_name)
        self.metric_relationships = {}
        self.load_existing_relationships()
    
    def load_existing_relationships(self):
        """Load existing relationships from database"""
        try:
            with self.db.connect() as conn:
                result = conn.execute(text("""
                    SELECT metric1, metric2, strength, co_occurrence_count 
                    FROM metric_relationships
                """))
                
                for row in result:
                    key = f"{row.metric1}_{row.metric2}"
                    self.metric_relationships[key] = QueryPattern(
                        metric1=row.metric1,
                        metric2=row.metric2,
                        strength=row.strength,
                        count=row.co_occurrence_count,
                        last_seen=datetime.utcnow()
                    )
            
            logger.info(f"Loaded {len(self.metric_relationships)} existing relationships")
        except Exception as e:
            logger.error(f"Error loading relationships: {e}")
    
    def analyze_recent_patterns(self, hours_back: int = 24):
        """Analyze patterns from recent queries"""
        try:
            with self.db.connect() as conn:
                result = conn.execute(text("""
                    SELECT session_id, business_entities, timestamp
                    FROM query_logs 
                    WHERE timestamp >= :since AND success = true
                    ORDER BY session_id, timestamp
                """), {'since': datetime.utcnow() - timedelta(hours=hours_back)})
                
                session_metrics = {}
                for row in result:
                    session_id = row.session_id
                    entities = json.loads(row.business_entities or '{}')
                    metrics = entities.get('metrics', [])
                    
                    if session_id not in session_metrics:
                        session_metrics[session_id] = set()
                    session_metrics[session_id].update(metrics)
                
                # Find co-occurrences
                new_relationships = []
                for session_id, metrics in session_metrics.items():
                    metrics_list = list(metrics)
                    for i, metric1 in enumerate(metrics_list):
                        for metric2 in metrics_list[i+1:]:
                            new_relationships.append((metric1, metric2))
                
                # Update relationship strengths
                self._update_relationships(new_relationships)
                
            logger.info(f"Analyzed patterns from {len(session_metrics)} sessions")
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
    
    def _update_relationships(self, relationships: List[tuple]):
        """Update relationship strengths and store in database"""
        relationship_counts = {}
        for metric1, metric2 in relationships:
            # Ensure consistent ordering
            if metric1 > metric2:
                metric1, metric2 = metric2, metric1
            
            key = f"{metric1}_{metric2}"
            relationship_counts[key] = relationship_counts.get(key, 0) + 1
        
        # Update database
        for key, count in relationship_counts.items():
            metric1, metric2 = key.split('_', 1)
            
            with self.db.connect() as conn:
                # Check if relationship exists
                existing = conn.execute(text("""
                    SELECT co_occurrence_count FROM metric_relationships 
                    WHERE metric1 = :m1 AND metric2 = :m2
                """), {'m1': metric1, 'm2': metric2}).fetchone()
                
                if existing:
                    new_count = existing.co_occurrence_count + count
                    strength = min(new_count / 10.0, 1.0)  # Cap at 1.0
                    
                    conn.execute(text("""
                        UPDATE metric_relationships 
                        SET co_occurrence_count = :count, strength = :strength, last_updated = :now
                        WHERE metric1 = :m1 AND metric2 = :m2
                    """), {
                        'count': new_count, 'strength': strength, 'now': datetime.utcnow(),
                        'm1': metric1, 'm2': metric2
                    })
                else:
                    strength = min(count / 10.0, 1.0)
                    conn.execute(text("""
                        INSERT INTO metric_relationships VALUES 
                        (:id, :m1, :m2, :strength, :count, :now)
                    """), {
                        'id': str(uuid.uuid4()), 'm1': metric1, 'm2': metric2,
                        'strength': strength, 'count': count, 'now': datetime.utcnow()
                    })
                
                conn.commit()
    
    def get_related_metrics(self, current_metrics: List[str], threshold: float = 0.3) -> List[QueryRecommendation]:
        """Get related metrics based on learned patterns"""
        recommendations = []
        
        for metric in current_metrics:
            with self.db.connect() as conn:
                result = conn.execute(text("""
                    SELECT metric1, metric2, strength, co_occurrence_count
                    FROM metric_relationships 
                    WHERE (metric1 = :metric OR metric2 = :metric) AND strength >= :threshold
                    ORDER BY strength DESC LIMIT 3
                """), {'metric': metric, 'threshold': threshold})
                
                for row in result:
                    related_metric = row.metric2 if row.metric1 == metric else row.metric1
                    recommendations.append(QueryRecommendation(
                        type='related_metric',
                        message=f"Based on {row.co_occurrence_count} similar analyses, you might also want to examine {related_metric}",
                        confidence=row.strength,
                        metadata={
                            'current_metric': metric,
                            'suggested_metric': related_metric,
                            'co_occurrences': row.co_occurrence_count
                        }
                    ))
        
        return recommendations

class RecommendationEngine:
    def __init__(self, metric_analyzer: BusinessMetricAnalyzer, db_engine):
        self.analyzer = metric_analyzer
        self.db = db_engine
        self.llm = OpenAI(temperature=0.3)
        
        self.recommendation_prompt = PromptTemplate(
            input_variables=["current_query", "related_metrics", "query_history"],
            template="""
            Based on the current query: "{current_query}"
            Related metrics often analyzed together: {related_metrics}
            Recent query pattern: {query_history}
            
            Provide 1-2 specific, actionable recommendations for additional analyses that would provide valuable business insights.
            Format each recommendation as a clear, concise suggestion.
            """
        )
        
        self.recommendation_chain = LLMChain(llm=self.llm, prompt=self.recommendation_prompt)
    
    def generate_recommendations(self, session_id: str, current_query: str, 
                               current_entities: Dict[str, List[str]]) -> List[QueryRecommendation]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Get metric-based recommendations
        current_metrics = current_entities.get('metrics', [])
        if current_metrics:
            metric_recs = self.analyzer.get_related_metrics(current_metrics)
            recommendations.extend(metric_recs)
        
        # Get query history for context
        query_history = self._get_recent_session_queries(session_id)
        
        # Generate LLM-based recommendations
        if current_metrics and query_history:
            try:
                llm_response = self.recommendation_chain.run(
                    current_query=current_query,
                    related_metrics=', '.join(current_metrics),
                    query_history=query_history
                )
                
                recommendations.append(QueryRecommendation(
                    type='context',
                    message=llm_response.strip(),
                    confidence=0.7,
                    metadata={'source': 'llm_analysis'}
                ))
            except Exception as e:
                logger.error(f"Error generating LLM recommendations: {e}")
        
        # Performance-based recommendations
        perf_recs = self._get_performance_recommendations(current_query)
        recommendations.extend(perf_recs)
        
        return recommendations[:3]  # Limit to top 3
    
    def _get_recent_session_queries(self, session_id: str, limit: int = 5) -> str:
        """Get recent queries from the same session"""
        try:
            with self.db.connect() as conn:
                result = conn.execute(text("""
                    SELECT natural_language_query FROM query_logs 
                    WHERE session_id = :session_id AND success = true
                    ORDER BY timestamp DESC LIMIT :limit
                """), {'session_id': session_id, 'limit': limit})
                
                queries = [row.natural_language_query for row in result]
                return ' -> '.join(reversed(queries))
        except Exception as e:
            logger.error(f"Error getting session queries: {e}")
            return ""
    
    def _get_performance_recommendations(self, current_query: str) -> List[QueryRecommendation]:
        """Generate performance-based recommendations"""
        recommendations = []
        
        # Check for common performance patterns
        if any(word in current_query.lower() for word in ['all time', 'entire', 'complete']):
            recommendations.append(QueryRecommendation(
                type='optimization',
                message="For better performance, consider filtering to a specific time range like 'last 6 months' or 'current year'",
                confidence=0.8,
                metadata={'optimization_type': 'time_filtering'}
            ))
        
        return recommendations

class QueryOptimizationAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize components
        self._initialize_components()
        
        # Setup scheduled jobs
        self._setup_scheduler()
        
        logger.info("Query Optimization Agent initialized")
    
    def _initialize_components(self):
        """Initialize all agent components"""
        # Database connection
        self.db_engine = create_engine(self.config['database']['connection_string'])
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            decode_responses=True
        ) if self.config.get('redis') else None
        
        # Pinecone setup
        if self.config.get('pinecone'):
            pinecone.init(
                api_key=self.config['pinecone']['api_key'],
                environment=self.config['pinecone']['environment']
            )
            self.pinecone_index = pinecone.Index(self.config['pinecone']['index_name'])
        else:
            self.pinecone_index = None
        
        # Initialize core components
        self.logger = QueryPatternLogger(self.config['database']['connection_string'])
        self.session_manager = SessionManager(self.db_engine, self.redis_client)
        self.metric_analyzer = BusinessMetricAnalyzer(self.pinecone_index, self.db_engine)
        self.recommendation_engine = RecommendationEngine(self.metric_analyzer, self.db_engine)
    
    def _setup_scheduler(self):
        """Setup scheduled jobs for pattern analysis"""
        # Analyze patterns every hour
        schedule.every().hour.do(self._hourly_pattern_analysis)
        
        # Daily comprehensive analysis
        schedule.every().day.at("02:00").do(self._daily_analysis)
        
        # Weekly cleanup
        schedule.every().week.do(self._weekly_cleanup)
    
    def start(self):
        """Start the agent"""
        self.running = True
        logger.info("Starting Query Optimization Agent")
        
        # Start background scheduler
        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        # Initial pattern analysis
        self.executor.submit(self._initial_analysis)
        
        logger.info("Query Optimization Agent started successfully")
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Query Optimization Agent stopped")
    
    def process_query(self, user_id: str, session_id: str, nl_query: str, 
                     generated_sql: str, execution_time: Optional[int] = None,
                     result_count: Optional[int] = None) -> List[QueryRecommendation]:
        """Process a query and return recommendations"""
        try:
            # Log the query
            query_id = self.logger.log_query(
                session_id=session_id,
                user_id=user_id,
                nl_query=nl_query,
                generated_sql=generated_sql,
                execution_time=execution_time,
                result_count=result_count,
                success=True
            )
            
            # Add to session
            self.session_manager.add_query_to_session(session_id, query_id)
            
            # Extract entities for recommendations
            entities = self.logger._extract_business_entities(nl_query)
            
            # Generate recommendations
            recommendations = self.recommendation_engine.generate_recommendations(
                session_id, nl_query, entities
            )
            
            logger.info(f"Processed query for user {user_id}, generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return []
    
    def create_session(self, user_id: str, session_type: str = 'ad-hoc') -> str:
        """Create a new user session"""
        return self.session_manager.start_session(user_id, session_type)
    
    def end_session(self, session_id: str):
        """End a user session"""
        self.session_manager.end_session(session_id)
    
    def _run_scheduler(self):
        """Run the background scheduler"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _hourly_pattern_analysis(self):
        """Hourly pattern analysis job"""
        logger.info("Running hourly pattern analysis")
        try:
            self.metric_analyzer.analyze_recent_patterns(hours_back=1)
        except Exception as e:
            logger.error(f"Error in hourly analysis: {e}")
    
    def _daily_analysis(self):
        """Daily comprehensive analysis"""
        logger.info("Running daily comprehensive analysis")
        try:
            self.metric_analyzer.analyze_recent_patterns(hours_back=24)
            # Add more comprehensive analysis here
        except Exception as e:
            logger.error(f"Error in daily analysis: {e}")
    
    def _weekly_cleanup(self):
        """Weekly maintenance and cleanup"""
        logger.info("Running weekly cleanup")
        try:
            # Clean up old sessions, optimize database, etc.
            with self.db_engine.connect() as conn:
                # Remove sessions older than 30 days
                conn.execute(text("""
                    DELETE FROM user_sessions 
                    WHERE start_time < :cutoff
                """), {'cutoff': datetime.utcnow() - timedelta(days=30)})
                conn.commit()
        except Exception as e:
            logger.error(f"Error in weekly cleanup: {e}")
    
    def _initial_analysis(self):
        """Initial analysis when agent starts"""
        logger.info("Running initial pattern analysis")
        try:
            self.metric_analyzer.analyze_recent_patterns(hours_back=168)  # Last week
        except Exception as e:
            logger.error(f"Error in initial analysis: {e}")

# Agent Factory and Configuration
def create_agent(config_path: str = None) -> QueryOptimizationAgent:
    """Factory function to create and configure the agent"""
    
    # Default configuration
    default_config = {
        'database': {
            'connection_string': 'sqlite:///query_optimization.db'  # Change to your DB
        },
        'redis': {
            'host': 'localhost',
            'port': 6379
        },
        'pinecone': {
            'api_key': 'your-pinecone-api-key',
            'environment': 'your-environment',
            'index_name': 'query-patterns'
        },
        'openai': {
            'api_key': 'your-openai-api-key'
        }
    }
    
    # Load custom config if provided
    if config_path:
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            default_config.update(custom_config)
    
    return QueryOptimizationAgent(default_config)

# Example usage and integration
async def main():
    """Example of how to start and use the agent"""
    
    # Create and start the agent
    agent = create_agent()
    agent.start()
    
    try:
        # Example usage
        user_id = "user123"
        session_id = agent.create_session(user_id, "dashboard")
        
        # Simulate query processing
        recommendations = agent.process_query(
            user_id=user_id,
            session_id=session_id,
            nl_query="Show me sales by region for last quarter",
            generated_sql="SELECT region, SUM(sales) FROM sales_data WHERE date >= '2024-01-01' GROUP BY region",
            execution_time=1500,
            result_count=12
        )
        
        print("Recommendations:")
        for rec in recommendations:
            print(f"- [{rec.type}] {rec.message} (confidence: {rec.confidence:.2f})")
        
        # Keep the agent running
        while True:
            await asyncio.sleep(60)  # Agent continues running in background
            
    except KeyboardInterrupt:
        logger.info("Shutting down agent...")
        agent.stop()

if __name__ == "__main__":
    asyncio.run(main())