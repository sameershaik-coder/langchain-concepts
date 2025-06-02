-- Query Log Table
CREATE TABLE query_logs (
    id UUID PRIMARY KEY,
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    timestamp TIMESTAMP,
    natural_language_query TEXT,
    generated_sql TEXT,
    sql_hash VARCHAR(64), -- For deduplication
    execution_time_ms INTEGER,
    result_row_count INTEGER,
    query_category VARCHAR(100), -- sales, marketing, finance, etc.
    business_entities ARRAY, -- extracted entities like 'revenue', 'customers'
    query_complexity_score INTEGER, -- 1-10 based on SQL complexity
    success BOOLEAN,
    error_message TEXT
);

-- Session Tracking Table
CREATE TABLE user_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_queries INTEGER,
    session_type VARCHAR(50) -- dashboard, ad-hoc, report_building
);

-- Query Sequences Table
CREATE TABLE query_sequences (
    id UUID PRIMARY KEY,
    session_id VARCHAR(255),
    query_order INTEGER,
    query_id UUID REFERENCES query_logs(id),
    time_gap_seconds INTEGER -- time since previous query
);