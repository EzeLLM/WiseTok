import React, { useState, useEffect, useCallback, useMemo, Fragment } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ChartDataPoint {
  timestamp: number;
  value: number;
  label: string;
}

interface AnalyticsProps {
  dataUrl: string;
  title: string;
  refreshInterval?: number;
  onDataChange?: (data: ChartDataPoint[]) => void;
}

const AnalyticsDashboard: React.FC<AnalyticsProps> = ({
  dataUrl,
  title,
  refreshInterval = 5000,
  onDataChange
}) => {
  const [data, setData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedRange, setSelectedRange] = useState<'day' | 'week' | 'month'>('week');

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${dataUrl}?range=${selectedRange}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      setData(json.points || []);
      onDataChange?.(json.points || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [dataUrl, selectedRange, onDataChange]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchData, refreshInterval]);

  const stats = useMemo(() => {
    if (data.length === 0) return { min: 0, max: 0, avg: 0 };
    const values = data.map(d => d.value);
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      avg: values.reduce((a, b) => a + b, 0) / values.length
    };
  }, [data]);

  const handleRangeChange = useCallback((newRange: typeof selectedRange) => {
    setSelectedRange(newRange);
  }, []);

  return (
    <div className="analytics-dashboard">
      <header className="dashboard-header">
        <h1 className="dashboard-title">{title}</h1>
        <div className="controls">
          {(['day', 'week', 'month'] as const).map(range => (
            <button
              key={range}
              className={`btn-range ${selectedRange === range ? 'active' : ''}`}
              onClick={() => handleRangeChange(range)}
            >
              {range.charAt(0).toUpperCase() + range.slice(1)}
            </button>
          ))}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <p>{error}</p>
          <button onClick={fetchData}>Retry</button>
        </div>
      )}

      {loading ? (
        <div className="spinner">Loading...</div>
      ) : (
        <>
          <div className="stats-grid">
            {Object.entries(stats).map(([key, val]) => (
              <Fragment key={key}>
                <div className="stat-card">
                  <span className="stat-label">{key.toUpperCase()}</span>
                  <span className="stat-value">{val.toFixed(2)}</span>
                </div>
              </Fragment>
            ))}
          </div>

          <ResponsiveContainer width="100%" height={400}>
            <LineChart
              data={data}
              margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 12 }}
              />
              <YAxis />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #ccc'
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#8884d8"
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
};

export default AnalyticsDashboard;
