from typing import Dict, List, Optional, Tuple
import time
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import snowflake.connector
import pandas as pd

class EnhancedNLToSQLPipeline:
    """Enhanced version of your existing pipeline with Query Optimization Agent"""
    
    def __init__(self, agent: QueryOptimizationAgent, snowflake_config: Dict, 
                 openai_api_key: str):
        self.agent = agent
        self.snowflake_config = snowflake_config
        
        # Your existing LangChain components
        self.llm = OpenAI(api_key=openai_api_key, temperature=0)
        self._setup_chains()
        
        # Snowflake connection
        self.snowflake_conn = None
        self._connect_snowflake()
    
    def _setup_chains(self):
        """Setup your existing LangChain chains"""
        
        # SQL Generation Chain
        sql_prompt = PromptTemplate(
            input_variables=["nl_query", "schema_info", "context"],
            template="""
            Given the database schema: {schema_info}
            Context from previous queries: {context}
            
            Convert this natural language query to SQL: {nl_query}
            
            Return only the SQL query without explanations.
            """
        )
        self.sql_chain = LLMChain(llm=self.llm, prompt=sql_prompt)
        
        # Insight Generation Chain
        insight_prompt = PromptTemplate(
            input_variables=["query_result", "original_query", "recommendations"],
            template="""
            Based on the query: {original_query}
            Query results: {query_result}
            
            Agent recommendations: {recommendations}
            
            Provide business insights and suggest follow-up analyses.
            """
        )
        self.insight_chain = LLMChain(llm=self.llm, prompt=insight_prompt)
    
    def _connect_snowflake(self):
        """Connect to Snowflake"""
        try:
            self.snowflake_conn = snowflake.connector.connect(**self.snowflake_config)
        except Exception as e:
            print(f"Snowflake connection error: {e}")
    
    def process_user_query(self, user_id: str, nl_query: str, 
                          session_id: Optional[str] = None) -> Dict:
        """Main pipeline that processes user queries with agent integration"""
        
        # Create session if not provided
        if not session_id:
            session_id = self.agent.create_session(user_id)
        
        start_time = time.time()
        
        try:
            # Step 1: Get schema context and query history
            schema_info = self._get_schema_context()
            context = self._get_session_context(session_id)
            
            # Step 2: Generate SQL using your existing chain
            generated_sql = self.sql_chain.run(
                nl_query=nl_query,
                schema_info=schema_info,
                context=context
            ).strip()
            
            # Step 3: Execute SQL query
            query_result, result_count = self._execute_snowflake_query(generated_sql)
            execution_time = int((time.time() - start_time) * 1000)
            
            # Step 4: Get recommendations from agent BEFORE returning results
            recommendations = self.agent.process_query(
                user_id=user_id,
                session_id=session_id,
                nl_query=nl_query,
                generated_sql=generated_sql,
                execution_time=execution_time,
                result_count=result_count
            )
            
            # Step 5: Generate insights with agent recommendations
            insights = self._generate_insights(nl_query, query_result, recommendations)
            
            # Step 6: Prepare visualizations if needed
            visualizations = self._prepare_visualizations(query_result, nl_query)
            
            return {
                'success': True,
                'session_id': session_id,
                'query': nl_query,
                'sql': generated_sql,
                'results': query_result,
                'insights': insights,
                'recommendations': [
                    {
                        'type': rec.type,
                        'message': rec.message,
                        'confidence': rec.confidence
                    } for rec in recommendations
                ],
                'visualizations': visualizations,
                'execution_time_ms': execution_time,
                'result_count': result_count
            }
            
        except Exception as e:
            # Log failed query to agent
            self.agent.logger.log_query(
                session_id=session_id,
                user_id=user_id,
                nl_query=nl_query,
                generated_sql=generated_sql if 'generated_sql' in locals() else '',
                success=False,
                error=str(e)
            )
            
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    def _get_schema_context(self) -> str:
        """Get relevant schema information for SQL generation"""
        # Your existing schema retrieval logic
        return """
        Tables: sales_data (date, region, product, sales, profit)
                customers (id, name, region, segment, signup_date)
                products (id, name, category, price, cost)
        """
    
    def _get_session_context(self, session_id: str) -> str:
        """Get context from previous queries in session"""
        try:
            with self.agent.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT natural_language_query, query_category 
                    FROM query_logs 
                    WHERE session_id = :session_id AND success = true
                    ORDER BY timestamp DESC LIMIT 3
                """), {'session_id': session_id})
                
                queries = [f"{row.query_category}: {row.natural_language_query}" 
                          for row in result]
                return "Previous queries: " + " | ".join(queries) if queries else ""
        except:
            return ""
    
    def _execute_snowflake_query(self, sql: str) -> Tuple[List[Dict], int]:
        """Execute SQL query on Snowflake"""
        try:
            cursor = self.snowflake_conn.cursor()
            cursor.execute(sql)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = [dict(zip(columns, row)) for row in rows]
            
            cursor.close()
            return results, len(results)
            
        except Exception as e:
            raise Exception(f"Snowflake query error: {e}")
    
    def _generate_insights(self, original_query: str, results: List[Dict], 
                          recommendations: List) -> str:
        """Generate insights using LLM with agent recommendations"""
        
        # Format results for LLM
        if len(results) > 5:
            formatted_results = str(results[:5]) + f"\n... and {len(results)-5} more rows"
        else:
            formatted_results = str(results)
        
        # Format recommendations
        rec_text = "\n".join([f"- {rec.message}" for rec in recommendations])
        
        try:
            insights = self.insight_chain.run(
                original_query=original_query,
                query_result=formatted_results,
                recommendations=rec_text
            )
            return insights
        except Exception as e:
            return f"Unable to generate insights: {e}"
    
    def _prepare_visualizations(self, results: List[Dict], query: str) -> Dict:
        """Prepare visualization suggestions based on query and results"""
        if not results:
            return {}
        
        # Analyze result structure
        numeric_columns = []
        categorical_columns = []
        
        for key, value in results[0].items():
            if isinstance(value, (int, float)):
                numeric_columns.append(key)
            else:
                categorical_columns.append(key)
        
        visualizations = {}
        
        # Suggest chart types based on data structure
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            visualizations['bar_chart'] = {
                'type': 'bar',
                'x_axis': categorical_columns[0],
                'y_axis': numeric_columns[0],
                'title': f"{numeric_columns[0]} by {categorical_columns[0]}"
            }
        
        if len(numeric_columns) >= 2:
            visualizations['scatter_plot'] = {
                'type': 'scatter',
                'x_axis': numeric_columns[0],
                'y_axis': numeric_columns[1],
                'title': f"{numeric_columns[1]} vs {numeric_columns[0]}"
            }
        
        return visualizations

class StoryboardEnhancer:
    """Enhanced storyboarding with agent recommendations"""
    
    def __init__(self, pipeline: EnhancedNLToSQLPipeline):
        self.pipeline = pipeline
        self.stories = {}  # session_id -> story data
    
    def create_story(self, user_id: str, title: str) -> str:
        """Create a new story/presentation"""
        session_id = self.pipeline.agent.create_session(user_id, 'storyboard')
        
        self.stories[session_id] = {
            'title': title,
            'user_id': user_id,
            'slides': [],
            'created_at': time.time()
        }
        
        return session_id
    
    def add_query_to_story(self, session_id: str, nl_query: str) -> Dict:
        """Add a query result as a slide to the story"""
        
        if session_id not in self.stories:
            raise ValueError("Story session not found")
        
        # Process query with full pipeline
        result = self.pipeline.process_user_query(
            user_id=self.stories[session_id]['user_id'],
            nl_query=nl_query,
            session_id=session_id
        )
        
        if result['success']:
            # Create slide from result
            slide = {
                'query': nl_query,
                'results': result['results'],
                'insights': result['insights'],
                'visualizations': result['visualizations'],
                'recommendations': result['recommendations'],
                'timestamp': time.time()
            }
            
            self.stories[session_id]['slides'].append(slide)
            
            # Generate story flow suggestions
            flow_suggestions = self._suggest_story_flow(session_id)
            result['story_suggestions'] = flow_suggestions
        
        return result
    
    def _suggest_story_flow(self, session_id: str) -> List[str]:
        """Suggest narrative flow based on current slides and agent patterns"""
        story = self.stories[session_id]
        
        if len(story['slides']) < 2:
            return ["Consider adding complementary metrics to build a complete narrative"]
        
        # Analyze slide sequence
        suggestions = []
        
        # Check for logical flow
        slide_categories = [self._categorize_slide(slide) for slide in story['slides']]
        
        if 'overview' not in slide_categories:
            suggestions.append("Consider starting with an overview or summary slide")
        
        if 'trend' not in slide_categories and len(story['slides']) > 2:
            suggestions.append("Add time-based trend analysis to show progression")
        
        # Use agent recommendations for flow
        latest_recs = story['slides'][-1].get('recommendations', [])
        for rec in latest_recs[:2]:  # Top 2 recommendations
            suggestions.append(f"Story flow: {rec['message']}")
        
        return suggestions[:3]  # Limit suggestions
    
    def _categorize_slide(self, slide: Dict) -> str:
        """Categorize slide content"""
        query = slide['query'].lower()
        
        if any(word in query for word in ['total', 'sum', 'overall', 'all']):
            return 'overview'
        elif any(word in query for word in ['trend', 'over time', 'monthly', 'quarterly']):
            return 'trend'
        elif any(word in query for word in ['by region', 'by product', 'by category']):
            return 'breakdown'
        else:
            return 'detail'
    
    def export_story(self, session_id: str) -> Dict:
        """Export complete story with narrative suggestions"""
        if session_id not in self.stories:
            raise ValueError("Story not found")
        
        story = self.stories[session_id]
        
        # Generate executive summary
        summary = self._generate_story_summary(story)
        
        return {
            'title': story['title'],
            'summary': summary,
            'slides': story['slides'],
            'slide_count': len(story['slides']),
            'created_at': story['created_at'],
            'narrative_flow': self._suggest_story_flow(session_id)
        }
    
    def _generate_story_summary(self, story: Dict) -> str:
        """Generate executive summary of the story"""
        if not story['slides']:
            return "No analysis completed yet."
        
        # Extract key insights from all slides
        insights = []
        for slide in story['slides']:
            if slide.get('insights'):
                insights.append(slide['insights'][:100] + "...")  # Truncate
        
        return f"Analysis covers {len(story['slides'])} key areas. " + " ".join(insights[:3])

# Configuration and startup
class AgentOrchestrator:
    """Main orchestrator that manages the entire system"""
    
    def __init__(self, config_file: str):
        self.config = self._load_config(config_file)
        self.agent = None
        self.pipeline = None
        self.storyboard = None
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def initialize(self):
        """Initialize all components"""
        print("Initializing Query Optimization Agent...")
        
        # Create and start the agent
        self.agent = QueryOptimizationAgent(self.config['agent'])
        self.agent.start()
        
        # Initialize enhanced pipeline
        self.pipeline = EnhancedNLToSQLPipeline(
            agent=self.agent,
            snowflake_config=self.config['snowflake'],
            openai_api_key=self.config['openai']['api_key']
        )
        
        # Initialize storyboard enhancer
        self.storyboard = StoryboardEnhancer(self.pipeline)
        
        print("All components initialized successfully!")
    
    def process_query(self, user_id: str, query: str, session_id: str = None) -> Dict:
        """Main entry point for query processing"""
        return self.pipeline.process_user_query(user_id, query, session_id)
    
    def create_story(self, user_id: str, title: str) -> str:
        """Create new storyboard"""
        return self.storyboard.create_story(user_id, title)
    
    def add_to_story(self, session_id: str, query: str) -> Dict:
        """Add query to storyboard"""
        return self.storyboard.add_query_to_story(session_id, query)
    
    def shutdown(self):
        """Gracefully shutdown all components"""
        if self.agent:
            self.agent.stop()
        print("System shutdown complete")

# Example usage
def main():
    """Example of complete system usage"""
    
    # Initialize the orchestrator
    orchestrator = AgentOrchestrator('config.json')
    orchestrator.initialize()
    
    try:
        # Example 1: Single query processing
        result = orchestrator.process_query(
            user_id="analyst_1",
            query="Show me sales performance by region for Q4"
        )
        
        print("Query Result:")
        print(f"SQL: {result['sql']}")
        print(f"Insights: {result['insights']}")
        print("Recommendations:")
        for rec in result['recommendations']:
            print(f"  - {rec['message']}")
        
        # Example 2: Storyboard creation
        story_id = orchestrator.create_story("analyst_1", "Q4 Performance Review")
        
        # Add multiple queries to build story
        queries = [
            "Overall sales performance for Q4",
            "Sales by region breakdown",
            "Profit margins by product category",
            "Customer acquisition trends"
        ]
        
        for query in queries:
            story_result = orchestrator.add_to_story(story_id, query)
            print(f"Added to story: {query}")
            if story_result.get('story_suggestions'):
                print("Story suggestions:", story_result['story_suggestions'])
        
        # Export completed story
        final_story = orchestrator.storyboard.export_story(story_id)
        print(f"Final story: {final_story['title']}")
        print(f"Summary: {final_story['summary']}")
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        orchestrator.shutdown()

if __name__ == "__main__":
    main()