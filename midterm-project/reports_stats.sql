WITH report_stats AS (
      SELECT
          r.id as report_id,
          COUNT(DISTINCT rb.block_id) as num_blocks,

          -- SQL blocks
          COUNT(DISTINCT CASE WHEN b.type = 'sql' THEN rb.block_id END) as num_sql_blocks,

          -- Writeback blocks
          COUNT(DISTINCT CASE WHEN b.type = 'writeback' THEN rb.block_id END) as num_writeback_blocks,

          -- Visualization blocks
          COUNT(DISTINCT CASE WHEN b.type IN ('plotly','kpi_card', 'table') THEN
  rb.block_id END) as num_viz_blocks,

          -- Tableau blocks
          COUNT(DISTINCT CASE WHEN b.type IN ('tableau', 'tableau_embed') THEN
  rb.block_id END) as tableau_blocks,

          -- Email blocks
          COUNT(DISTINCT CASE WHEN b.type IN ('email') THEN rb.block_id END) as
  num_email_blocks,

          -- Slack blocks
          COUNT(DISTINCT CASE WHEN b.type IN ('slack') THEN rb.block_id END) as
  num_slack_blocks,

          -- Api blocks
          COUNT(DISTINCT CASE WHEN b.type IN ('api_request') THEN rb.block_id END) as
  num_api_blocks,

          -- Storage blocks
          COUNT(DISTINCT CASE WHEN b.type IN ('aws_s3', 'azure_storage_blob', 'gcp_bucket', 'gdrive', 'sftp') THEN
  rb.block_id END) as num_storage_blocks,

          -- Control flow blocks
          COUNT(DISTINCT CASE WHEN b.type IN ('if_start', 'if_end', 'else_', 'for_start', 'for_end') THEN rb.block_id
   END) as num_control_blocks,

          -- Parameters
          COUNT(DISTINCT CASE WHEN b.type = 'parameter' THEN rb.block_id END) as num_parameters,

          -- Extract unique databases from content_json
          COUNT(DISTINCT
              CASE
                  WHEN b.type IN ('sql', 'writeback')
                      AND b.content_json IS NOT NULL
                      AND b.content_json::jsonb != 'null'::jsonb
                      AND jsonb_typeof(b.content_json::jsonb) = 'object'
                  THEN b.content_json::jsonb ->> 'database_id'
              END
          ) as num_databases

      FROM report r
      LEFT JOIN report_block rb ON r.id = rb.report_id
      LEFT JOIN block b ON rb.block_id = b.id
      GROUP BY r.id
  ),

  -- Step 2: Get historical failure statistics (using data older than 7 days)
  historical_failures AS (
      SELECT
          nr.report_id,
          COUNT(*) FILTER (WHERE nr.status = 'failed') as historical_failure_count,
          COUNT(*) as total_executions,
          CASE
              WHEN COUNT(*) > 0 THEN
                  COUNT(*) FILTER (WHERE nr.status = 'failed')::FLOAT / COUNT(*)
              ELSE 0
          END as historical_failure_rate,
          MAX(nr.end_time) FILTER (WHERE nr.status = 'success') as last_success_time,
          AVG(EXTRACT(EPOCH FROM (nr.end_time - nr.start_time))) FILTER (WHERE nr.status = 'success' AND nr.end_time
  IS NOT NULL) as avg_duration_seconds
      FROM noteflow_run nr
      WHERE nr.created_on < CURRENT_DATE - INTERVAL '7 days'
          AND nr.status IN ('success', 'failed')
      GROUP BY nr.report_id
  ),

  -- Step 3: Extract database types for each execution
  report_databases AS (
      SELECT DISTINCT
          r.id as report_id,
          split_part(split_part(db.sqlalchemy_uri, '://', 1), '+', 1) as database_type
      FROM report r
      JOIN report_block rb ON r.id = rb.report_id
      JOIN block b ON rb.block_id = b.id
      JOIN dbs db ON db.id::text = b.content_json::jsonb ->> 'database_id'
      WHERE b.type IN ('sql', 'writeback')
          AND b.content_json IS NOT NULL
          AND b.content_json::jsonb != 'null'::jsonb
          AND jsonb_typeof(b.content_json::jsonb) = 'object'
          AND b.content_json::jsonb ->> 'database_id' IS NOT NULL
  )

  -- Final SELECT: Combine all features
  SELECT
      -- Target variable
      CASE WHEN nr.status = 'failed' THEN 1 ELSE 0 END as failed,

      -- Execution metadata (anonymized)
      nr.id as execution_id,
      MD5(nr.report_id::text) as report_hash,
      MD5(COALESCE(nr.workspace_id::text, 'null')) as workspace_hash,

      -- Report complexity features
      COALESCE(rs.num_blocks, 0) as num_blocks,
      COALESCE(rs.num_sql_blocks, 0) as num_sql_blocks,
      COALESCE(rs.num_writeback_blocks, 0) as num_writeback_blocks,
      COALESCE(rs.num_viz_blocks, 0) as num_viz_blocks,
      COALESCE(rs.num_integration_blocks, 0) as num_integration_blocks,
      COALESCE(rs.num_storage_blocks, 0) as num_storage_blocks,
      COALESCE(rs.num_control_blocks, 0) as num_control_blocks,
      COALESCE(rs.num_parameters, 0) as num_parameters,
      COALESCE(rs.num_databases, 0) as num_databases,

      -- Derived complexity score
      (COALESCE(rs.num_sql_blocks, 0) * 2 +
       COALESCE(rs.num_writeback_blocks, 0) * 3 +
       COALESCE(rs.num_viz_blocks, 0) +
       COALESCE(rs.num_integration_blocks, 0) * 2 +
       COALESCE(rs.num_control_blocks, 0) * 1.5)::INTEGER as complexity_score,

      -- Historical performance features
      COALESCE(hf.historical_failure_count, 0) as historical_failure_count,
      COALESCE(hf.total_executions, 0) as historical_executions,
      COALESCE(hf.historical_failure_rate, 0) as historical_failure_rate,
      COALESCE(hf.avg_duration_seconds, 0) as avg_historical_duration,

      -- Time since last success
      CASE
          WHEN hf.last_success_time IS NOT NULL
          THEN EXTRACT(EPOCH FROM (nr.start_time - hf.last_success_time)) / 3600
          ELSE NULL
      END as hours_since_last_success,

      -- Timing features
      EXTRACT(HOUR FROM nr.start_time) as hour_of_day,
      EXTRACT(DOW FROM nr.start_time) as day_of_week,
      CASE WHEN EXTRACT(DOW FROM nr.start_time) IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
      CASE WHEN EXTRACT(HOUR FROM nr.start_time) BETWEEN 9 AND 17 THEN 1 ELSE 0 END as is_business_hours,

      -- Execution context
      CASE WHEN nr.parent_id IS NOT NULL THEN 1 ELSE 0 END as is_sub_report,
      CASE WHEN nr.workflow_log_id IS NOT NULL THEN 1 ELSE 0 END as is_scheduled,

      -- Database types (will be one-hot encoded later)
      COALESCE(STRING_AGG(DISTINCT rd.database_type, ','), 'none') as database_types,

      -- Actual duration for this execution (if successful)
      CASE
          WHEN nr.status = 'success' AND nr.end_time IS NOT NULL AND nr.start_time IS NOT NULL
          THEN EXTRACT(EPOCH FROM (nr.end_time - nr.start_time))
          ELSE NULL
      END as duration_seconds,

      -- Execution date (for train/test split later)
      DATE(nr.start_time) as execution_date

  FROM noteflow_run nr
  LEFT JOIN report_stats rs ON nr.report_id = rs.report_id
  LEFT JOIN historical_failures hf ON nr.report_id = hf.report_id
  LEFT JOIN report_databases rd ON nr.report_id = rd.report_id

  WHERE
      nr.start_time IS NOT NULL
      AND nr.status IN ('success', 'failed')
      AND nr.created_on >= CURRENT_DATE - INTERVAL '6 months'
      AND nr.report_id IS NOT NULL

  GROUP BY
      nr.id, nr.report_id, nr.status, nr.start_time, nr.end_time,
      nr.parent_id, nr.workflow_log_id, nr.workspace_id,
      rs.num_blocks, rs.num_sql_blocks, rs.num_writeback_blocks,
      rs.num_viz_blocks, rs.num_integration_blocks, rs.num_storage_blocks,
      rs.num_control_blocks, rs.num_parameters, rs.num_databases,
      hf.historical_failure_count, hf.total_executions,
      hf.historical_failure_rate, hf.last_success_time, hf.avg_duration_seconds

  ORDER BY nr.start_time DESC
  LIMIT 100000;