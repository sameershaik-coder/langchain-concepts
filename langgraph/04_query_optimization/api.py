#!/usr/bin/env python3
"""
Query Optimization Agent - Startup Script (FastAPI Version)
This script initializes and runs the complete Query Optimization Agent system
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from query_optimization_agent import QueryOptimizationAgent, create_agent
from langchain_integration import AgentOrchestrator

def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_file = config.get('logging', {}).get('file', 'query_agent.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_config(config: dict) -> bool:
    """Validate configuration before starting"""
    required_sections = ['agent', 'snowflake', 'openai']
    
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required configuration section: {section}")
            return False
    
    # Validate database connection
    if 'connection_string' not in config['agent']['database']:
        print("Error: Database connection string is required")
        return False
    
    # Validate API keys
    if not config['openai'].get('api_key'):
        print("Error: OpenAI API key is required")
        return False
    
    return True

def create_sample_config(config_path: str):
    """Create a sample configuration file"""
    sample_config = {
        "agent": {
            "database": {
                "connection_string": "postgresql://username:password@localhost:5432/analytics_db"
            },
            "redis": {
                "host": "localhost",
                "port": 6379
            },
            "pinecone": {
                "api_key": "your-pinecone-api-key",
                "environment": "us-west1-gcp-free",
                "index_name": "query-patterns"
            }
        },
        "snowflake": {
            "user": "your_snowflake_user",
            "password": "your_snowflake_password",
            "account": "your_account.region",
            "warehouse": "COMPUTE_WH",
            "database": "ANALYTICS_DB",
            "schema": "PUBLIC"
        },
        "openai": {
            "api_key": "sk-your-openai-api-key"
        },
        "logging": {
            "level": "INFO",
            "file": "query_agent.log"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"Sample configuration created at {config_path}")
    print("Please edit the configuration file with your actual credentials and settings.")

class AgentService:
    """Service wrapper for the Query Optimization Agent"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.orchestrator = None
        self.running = False
    
    async def start(self):
        """Start the agent service"""
        try:
            # Load configuration
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Validate configuration
            if not validate_config(config):
                return False
            
            # Setup logging
            setup_logging(config)
            
            logger = logging.getLogger(__name__)
            logger.info("Starting Query Optimization Agent Service")
            
            # Initialize orchestrator
            self.orchestrator = AgentOrchestrator(self.config_path)
            await asyncio.get_event_loop().run_in_executor(None, self.orchestrator.initialize)
            
            self.running = True
            logger.info("Query Optimization Agent Service started successfully")
            
            return True
            
        except Exception as e:
            print(f"Failed to start service: {e}")
            return False
    
    async def stop(self):
        """Stop the agent service"""
        if self.orchestrator:
            await asyncio.get_event_loop().run_in_executor(None, self.orchestrator.shutdown)
        self.running = False
        logging.getLogger(__name__).info("Query Optimization Agent Service stopped")
    
    async def health_check(self) -> dict:
        """Check service health"""
        if not self.running or not self.orchestrator:
            return {"status": "stopped", "timestamp": datetime.now().isoformat()}
        
        try:
            # Test database connection
            def test_db():
                with self.orchestrator.agent.db_engine.connect() as conn:
                    conn.execute("SELECT 1")
            
            await asyncio.get_event_loop().run_in_executor(None, test_db)
            db_status = "healthy"
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        return {
            "status": "running",
            "database": db_status,
            "agent_running": self.orchestrator.agent.running,
            "timestamp": datetime.now().isoformat()
        }
    
    async def process_query(self, user_id: str, query: str, session_id: str = None) -> dict:
        """Process a user query"""
        if not self.running:
            return {"error": "Service not running"}
        
        # Run the blocking operation in a thread pool
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            self.orchestrator.process_query, 
            user_id, 
            query, 
            session_id
        )

# FastAPI Server
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    
    # Pydantic models for request/response validation
    class QueryRequest(BaseModel):
        user_id: str
        query: str
        session_id: Optional[str] = None
    
    class StoryCreateRequest(BaseModel):
        user_id: str
        title: str
    
    class StoryAddRequest(BaseModel):
        query: str
    
    class HealthResponse(BaseModel):
        status: str
        database: Optional[str] = None
        agent_running: Optional[bool] = None
        timestamp: str
    
    class QueryResponse(BaseModel):
        success: Optional[bool] = None
        sql: Optional[str] = None
        results: Optional[list] = None
        insights: Optional[str] = None
        recommendations: Optional[list] = None
        session_id: Optional[str] = None
        error: Optional[str] = None
    
    class StoryCreateResponse(BaseModel):
        story_id: str
    
    def create_api_server(agent_service: AgentService) -> FastAPI:
        """Create FastAPI server for the agent"""
        app = FastAPI(
            title="Query Optimization Agent API",
            description="API for the Query Optimization Agent system",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure this for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint"""
            return await agent_service.health_check()
        
        @app.post("/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest):
            """Process a user query"""
            try:
                result = await agent_service.process_query(
                    user_id=request.user_id,
                    query=request.query,
                    session_id=request.session_id
                )
                return QueryResponse(**result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/story/create", response_model=StoryCreateResponse)
        async def create_story(request: StoryCreateRequest):
            """Create a new story"""
            try:
                def create_story_sync():
                    return agent_service.orchestrator.create_story(
                        request.user_id, 
                        request.title
                    )
                
                story_id = await asyncio.get_event_loop().run_in_executor(
                    None, create_story_sync
                )
                return StoryCreateResponse(story_id=story_id)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/story/{story_id}/add")
        async def add_to_story(story_id: str, request: StoryAddRequest):
            """Add a query to an existing story"""
            try:
                def add_to_story_sync():
                    return agent_service.orchestrator.add_to_story(
                        story_id, 
                        request.query
                    )
                
                result = await asyncio.get_event_loop().run_in_executor(
                    None, add_to_story_sync
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.on_event("startup")
        async def startup_event():
            """Initialize the agent service on startup"""
            success = await agent_service.start()
            if not success:
                raise RuntimeError("Failed to start agent service")
        
        @app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            await agent_service.stop()
        
        return app
    
except ImportError:
    print("FastAPI not installed. API server will not be available.")
    print("Install with: pip install fastapi uvicorn")
    def create_api_server(agent_service):
        return None

async def run_test_queries(service: AgentService):
    """Run test queries asynchronously"""
    print("Running test queries...")
    test_queries = [
        "Show me total sales for last quarter",
        "What are the top 5 products by revenue?",
        "Sales performance by region",
        "Customer churn rate analysis"
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        result = await service.process_query("test_user", query)
        if result.get('success'):
            print(f"✓ SQL Generated: {result['sql'][:100]}...")
            print(f"✓ Recommendations: {len(result.get('recommendations', []))}")
        else:
            print(f"✗ Error: {result.get('error')}")

async def interactive_mode(service: AgentService):
    """Run interactive mode asynchronously"""
    print("Query Optimization Agent is running!")
    print("Type 'quit' to exit, 'health' to check status")
    
    user_id = "interactive_user"
    session_id = None
    
    while True:
        try:
            # Note: input() is blocking, but for demo purposes we'll keep it simple
            # In a real async application, you'd want to use aioconsole or similar
            user_input = input("\nEnter query: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'health':
                health = await service.health_check()
                print(f"Status: {health['status']}")
                print(f"Database: {health.get('database', 'unknown')}")
                continue
            elif not user_input:
                continue
            
            # Process query
            result = await service.process_query(user_id, user_input, session_id)
            
            if result.get('success'):
                session_id = result['session_id']  # Maintain session
                print(f"\n✓ Results: {len(result.get('results', []))} rows")
                print(f"✓ Insights: {result.get('insights', '')[:200]}...")
                
                if result.get('recommendations'):
                    print("\n📋 Recommendations:")
                    for i, rec in enumerate(result['recommendations'], 1):
                        print(f"  {i}. {rec['message']}")
            else:
                print(f"✗ Error: {result.get('error')}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Query Optimization Agent (FastAPI)')
    parser.add_argument('--config', '-c', default='config.json', 
                       help='Configuration file path')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a sample configuration file')
    parser.add_argument('--api', action='store_true',
                       help='Start FastAPI server')
    parser.add_argument('--host', default='0.0.0.0',
                       help='API server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                       help='API server port (default: 8000)')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--test', action='store_true',
                       help='Run test queries')
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config(args.config)
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Use --create-config to create a sample configuration file")
        return
    
    # Initialize service
    service = AgentService(args.config)
    
    try:
        if args.api:
            # Start FastAPI server
            app = create_api_server(service)
            if app:
                print(f"Starting FastAPI server on {args.host}:{args.port}")
                print(f"Health check: http://{args.host}:{args.port}/health")
                print(f"API docs: http://{args.host}:{args.port}/docs")
                print(f"Query endpoint: POST http://{args.host}:{args.port}/query")
                
                config = uvicorn.Config(
                    app=app,
                    host=args.host,
                    port=args.port,
                    reload=args.reload,
                    log_level="info"
                )
                server = uvicorn.Server(config)
                await server.serve()
            else:
                print("FastAPI not available. Cannot start API server.")
                print("Install with: pip install fastapi uvicorn")
        
        else:
            # Non-API modes need to start service manually
            if not await service.start():
                print("Failed to start service")
                return
            
            if args.test:
                await run_test_queries(service)
            else:
                await interactive_mode(service)
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\nShutting down...")
        await service.stop()

def sync_main():
    """Synchronous wrapper for the async main function"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")

if __name__ == "__main__":
    sync_main()