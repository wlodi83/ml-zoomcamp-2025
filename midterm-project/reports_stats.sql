WITH report_stats AS (
      SELECT
          r.id AS report_id,
          COUNT(DISTINCT rb.block_id) AS num_blocks,
          COUNT(DISTINCT CASE WHEN b.type = 'sql' THEN rb.block_id END) AS num_sql_blocks,
          COUNT(DISTINCT CASE WHEN b.type = 'writeback' THEN rb.block_id END) AS num_writeback_blocks,
          COUNT(DISTINCT CASE WHEN b.type IN ('plotly', 'kpi_card', 'table') THEN rb.block_id END) AS num_viz_blocks,
          COUNT(DISTINCT CASE WHEN b.type IN ('tableau', 'tableau_embed') THEN rb.block_id END) AS
  num_tableau_blocks,
          COUNT(DISTINCT CASE WHEN b.type = 'email' THEN rb.block_id END) AS num_email_blocks,
          COUNT(DISTINCT CASE WHEN b.type = 'slack' THEN rb.block_id END) AS num_slack_blocks,
          COUNT(DISTINCT CASE WHEN b.type = 'api_request' THEN rb.block_id END) AS num_api_blocks,
          COUNT(DISTINCT CASE WHEN b.type IN ('aws_s3', 'azure_storage_blob', 'gcp_bucket', 'gdrive') THEN
  rb.block_id END) AS num_storage_blocks,
          COUNT(DISTINCT CASE WHEN b.type = 'sftp' THEN rb.block_id END) AS num_sftp_blocks,
          COUNT(DISTINCT CASE WHEN b.type IN ('if_start', 'if_end', 'else_', 'for_start', 'for_end') THEN rb.block_id
   END) AS num_control_blocks,
          COUNT(DISTINCT CASE WHEN b.type = 'parameter' THEN rb.block_id END) AS num_parameters,
          COUNT(DISTINCT CASE
              WHEN b.type IN ('sql', 'writeback')
                AND b.content_json IS NOT NULL
                AND b.content_json::jsonb != 'null'::jsonb
                AND jsonb_typeof(b.content_json::jsonb) = 'object'
              THEN b.content_json::jsonb ->> 'database_id'
          END) AS num_databases
      FROM report r
          LEFT JOIN report_block rb ON r.id = rb.report_id
          LEFT JOIN block b ON rb.block_id = b.id
      --WHERE r.workspace_id NOT IN (8, 103, 104, 128, 338, 3104, 3121, 3122, 987)
      GROUP BY r.id
  ),

  report_databases AS (
      SELECT DISTINCT
          r.id AS report_id,
          split_part(split_part(db.sqlalchemy_uri, '://', 1), '+', 1) AS database_type
      FROM report r
          JOIN report_block rb ON r.id = rb.report_id
          JOIN block b ON rb.block_id = b.id
          JOIN dbs db ON db.uid::text = b.content_json::jsonb ->> 'database_id'
      WHERE b.type IN ('sql', 'writeback')
        AND b.content_json IS NOT NULL
        AND b.content_json::jsonb != 'null'::jsonb
        AND jsonb_typeof(b.content_json::jsonb) = 'object'
        AND b.content_json::jsonb ->> 'database_id' IS NOT NULL
        AND r.workspace_id NOT IN (8, 103, 104, 128, 338, 3104, 3121, 3122, 987)
  ),
  execution_history AS (
      SELECT
          id,
          report_id,
          status,
          start_time,
          end_time,
          COUNT(*) FILTER (WHERE status = 'failed')
              OVER (PARTITION BY report_id ORDER BY start_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
              AS historical_failure_count,

          COUNT(*) FILTER (WHERE status IN ('success', 'failed'))
              OVER (PARTITION BY report_id ORDER BY start_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
              AS historical_executions,

          AVG(EXTRACT(EPOCH FROM (end_time - start_time))) FILTER (WHERE status = 'success' AND end_time IS NOT NULL)
              OVER (PARTITION BY report_id ORDER BY start_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
              AS avg_historical_duration,

          MAX(end_time) FILTER (WHERE status = 'success')
              OVER (PARTITION BY report_id ORDER BY start_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
              AS last_success_time

      FROM noteflow_run
      WHERE start_time IS NOT NULL
        AND status IN ('success', 'failed')
        AND created_on >= CURRENT_DATE - INTERVAL '6 months'
        AND report_id IS NOT NULL
        --AND workspace_id NOT IN (8, 103, 104, 128, 338, 3104, 3121, 3122, 987)
  )

  SELECT
      -- Target variable
      CASE WHEN nr.status = 'failed' THEN 1 ELSE 0 END AS failed,

      -- Execution metadata
      nr.id AS execution_id,
      MD5(nr.report_id::text) AS report_hash,
      MD5(COALESCE(nr.workspace_id::text, 'null')) AS workspace_hash,

      -- Report features
      COALESCE(rs.num_blocks, 0) AS num_blocks,
      COALESCE(rs.num_sql_blocks, 0) AS num_sql_blocks,
      COALESCE(rs.num_writeback_blocks, 0) AS num_writeback_blocks,
      COALESCE(rs.num_viz_blocks, 0) AS num_viz_blocks,
      COALESCE(rs.num_tableau_blocks, 0) AS num_tableau_blocks,
      COALESCE(rs.num_email_blocks, 0) AS num_email_blocks,
      COALESCE(rs.num_slack_blocks, 0) AS num_slack_blocks,
      COALESCE(rs.num_api_blocks, 0) AS num_api_blocks,
      COALESCE(rs.num_sftp_blocks, 0) AS num_sftp_blocks,
      COALESCE(rs.num_storage_blocks, 0) AS num_storage_blocks,
      COALESCE(rs.num_control_blocks, 0) AS num_control_blocks,
      COALESCE(rs.num_parameters, 0) AS num_parameters,
      COALESCE(rs.num_databases, 0) AS num_databases,

      -- Historical performance
      COALESCE(eh.historical_failure_count, 0) AS historical_failure_count,
      COALESCE(eh.historical_executions, 0) AS historical_executions,

      -- Calculate failure rate
      CASE
          WHEN COALESCE(eh.historical_executions, 0) > 0
          THEN COALESCE(eh.historical_failure_count, 0)::FLOAT / eh.historical_executions
          ELSE 0
      END AS historical_failure_rate,

      COALESCE(eh.avg_historical_duration, 0) AS avg_historical_duration,

      -- Hours since last success
      CASE
          WHEN eh.last_success_time IS NOT NULL AND eh.last_success_time < nr.start_time
          THEN EXTRACT(EPOCH FROM (nr.start_time - eh.last_success_time)) / 3600
          ELSE NULL
      END AS hours_since_last_success,

      -- Timing features
      EXTRACT(HOUR FROM nr.start_time) AS hour_of_day,
      EXTRACT(DOW FROM nr.start_time) AS day_of_week,
      CASE WHEN EXTRACT(DOW FROM nr.start_time) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
      CASE WHEN EXTRACT(HOUR FROM nr.start_time) BETWEEN 9 AND 17 THEN 1 ELSE 0 END AS is_business_hours,

      -- Execution context
      CASE WHEN nr.parent_id IS NOT NULL THEN 1 ELSE 0 END AS is_rerun,
      CASE WHEN nr.workflow_log_id IS NOT NULL THEN 1 ELSE 0 END AS is_scheduled,

      -- Database types
      COALESCE(STRING_AGG(DISTINCT rd.database_type, ','), 'none') AS database_types,

      -- Actual duration
      CASE
          WHEN nr.status = 'success' AND nr.end_time IS NOT NULL AND nr.start_time IS NOT NULL
          THEN EXTRACT(EPOCH FROM (nr.end_time - nr.start_time))
          ELSE NULL
      END AS duration_seconds,

      -- Execution date
      DATE(nr.start_time) AS execution_date

  FROM noteflow_run nr
      LEFT JOIN report_stats rs ON nr.report_id = rs.report_id
      LEFT JOIN execution_history eh ON nr.id = eh.id
      LEFT JOIN report_databases rd ON nr.report_id = rd.report_id

  WHERE nr.start_time IS NOT NULL
    AND nr.status IN ('success', 'failed')
    AND nr.created_on >= CURRENT_DATE - INTERVAL '6 months'
    AND nr.report_id IS NOT NULL

  GROUP BY
      nr.id,
      nr.report_id,
      nr.status,
      nr.start_time,
      nr.end_time,
      nr.parent_id,
      nr.workflow_log_id,
      nr.workspace_id,
      rs.num_blocks,
      rs.num_sql_blocks,
      rs.num_writeback_blocks,
      rs.num_viz_blocks,
      rs.num_tableau_blocks,
      rs.num_email_blocks,
      rs.num_slack_blocks,
      rs.num_api_blocks,
      rs.num_sftp_blocks,
      rs.num_storage_blocks,
      rs.num_control_blocks,
      rs.num_parameters,
      rs.num_databases,
      eh.historical_failure_count,
      eh.historical_executions,
      eh.avg_historical_duration,
      eh.last_success_time

  ORDER BY nr.start_time DESC
  LIMIT 500000;