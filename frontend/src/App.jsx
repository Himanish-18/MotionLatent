import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import {
  Activity, ArrowDownRight, ArrowUpRight, BarChart3, BrainCircuit,
  CheckCircle2, ChevronDown, CircleHelp, Cpu, Database, Download,
  FileBarChart, Gauge, GitCompareArrows, LayoutDashboard, Layers3,
  Menu, RefreshCw, Search, ShieldCheck, Sparkles, Zap
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";

const METHOD_META = {
  "PCA": { label: "PCA", color: "#8b9cff", family: "Linear · orthogonal projection" },
  "FastICA": { label: "FastICA", color: "#42d6c8", family: "Linear · independent components" },
  "Autoencoder": { label: "MLP-AE", color: "#c19aff", family: "Neural · dense bottleneck" },
  "Conv1D-AE": { label: "Conv1D-AE", color: "#f4b86a", family: "Neural · temporal convolution" },
  "DSC-AE": { label: "DSC-AE", color: "#f28ca8", family: "Neural · depthwise separable" }
};
const METHOD_ORDER = ["PCA", "FastICA", "Autoencoder", "Conv1D-AE", "DSC-AE"];
const DIMENSIONS = [2, 4, 8, 16, 32, 64, 128];
const NAV = [
  { id: "overview", label: "Overview", icon: LayoutDashboard, group: "WORKSPACE" },
  { id: "models", label: "Model explorer", icon: Layers3 },
  { id: "performance", label: "Performance", icon: BarChart3 },
  { id: "edge", label: "Edge deployment", icon: Cpu },
  { id: "compare", label: "Compare models", icon: GitCompareArrows },
  { id: "dataset", label: "Dataset", icon: Database }
];

const numberOrNull = (value) => {
  if (value === null || value === undefined || value === "" || value === "N/A") return null;
  const parsed = Number.parseFloat(String(value).replace(/x$/i, ""));
  return Number.isFinite(parsed) ? parsed : null;
};
const percent = (value) => value == null ? null : value * 100;
const pretty = (value, digits = 2) => value == null ? "—" : value.toFixed(digits);
const methodLabel = (method) => METHOD_META[method]?.label || method;

function normalizeCsv(csvText) {
  const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true, dynamicTyping: false });
  if (parsed.errors.length) console.warn("CSV parsing warnings:", parsed.errors);
  return parsed.data.map((r) => ({
    method: r.Method?.trim(),
    dimension: numberOrNull(r["Latent Dimension"]),
    compression: numberOrNull(r["Compression Ratio"]),
    bytes: numberOrNull(r["Representation Bytes/Win"]),
    accuracy: percent(numberOrNull(r["Bal Acc (8-sub Macro)"])),
    accuracy9: percent(numberOrNull(r["Bal Acc (9-sub Macro)"])),
    accuracyStd: numberOrNull(r["Bal Acc Std (9-sub)"]),
    weightedAccuracy: percent(numberOrNull(r["Bal Acc Weighted"])),
    pooledAccuracy: percent(numberOrNull(r["True Pooled Bal Acc"])),
    macroF1: percent(numberOrNull(r["Macro-F1 (8-sub)"])),
    variance: percent(numberOrNull(r["Cumulative Explained Variance"])),
    trustworthiness: numberOrNull(r["Trustworthiness (k=5)"]),
    overlap: numberOrNull(r["k-NN Overlap (k=5)"]),
    mse: numberOrNull(r["Reconstruction MSE"]),
    rmse: numberOrNull(r["Reconstruction RMSE"]),
    flash: numberOrNull(r["Model Flash Size (KB)"]),
    latency: numberOrNull(r["Latency (ms/win)"]),
    evidence: r["Evidence Status"] || "Not specified"
  })).filter((r) => r.method);
}

function MetricCard({ label, value, detail, icon: Icon, accent = "blue", trend }) {
  return (
    <article className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        <span className={`metric-icon ${accent}`}><Icon size={17} /></span>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-foot">
        <span>{detail}</span>
        {trend && <span className="metric-trend"><ArrowUpRight size={13} />{trend}</span>}
      </div>
    </article>
  );
}

function Panel({ title, subtitle, action, children, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

function EmptyState({ message }) {
  return <div className="empty-state"><CircleHelp size={20} /><span>{message}</span></div>;
}

function App() {
  const [rows, setRows] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activePage, setActivePage] = useState("overview");
  const [mobileNav, setMobileNav] = useState(false);
  const [methodFilter, setMethodFilter] = useState("All methods");
  const [dimensionFilter, setDimensionFilter] = useState("All dimensions");
  const [search, setSearch] = useState("");
  const [compareDimension, setCompareDimension] = useState("32");
  const [compareA, setCompareA] = useState("PCA");
  const [compareB, setCompareB] = useState("DSC-AE");

  async function loadData() {
    setLoading(true);
    setError("");
    try {
      const [csvResponse, summaryResponse] = await Promise.all([
        fetch("/data/phase5_comparison.csv"),
        fetch("/data/phase5_summary.json")
      ]);
      if (!csvResponse.ok) throw new Error("Could not load phase5_comparison.csv. Check that it exists in public/data.");
      if (!summaryResponse.ok) throw new Error("Could not load phase5_summary.json. Check that it exists in public/data.");
      const [csvText, summaryJson] = await Promise.all([csvResponse.text(), summaryResponse.json()]);
      const normalized = normalizeCsv(csvText);
      if (!normalized.length) throw new Error("The benchmark CSV loaded, but it contained no usable records.");
      setRows(normalized);
      setSummary(summaryJson);
    } catch (err) {
      setError(err.message || "An unexpected error occurred while loading benchmark data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadData(); }, []);

  const candidates = useMemo(() => rows.filter((r) => r.method !== "Raw Baseline"), [rows]);
  const bestRecord = useMemo(
    () => [...candidates].filter((r) => r.accuracy != null).sort((a, b) => b.accuracy - a.accuracy)[0],
    [candidates]
  );
  const raw = rows.find((r) => r.method === "Raw Baseline");
  const pcaWinner = summary?.mvd_winners?.algorithmic_mvd_winner;
  const neuralWinner = summary?.mvd_winners?.ultra_constrained_neural_alternative;

  const accuracyChart = useMemo(() => DIMENSIONS.map((dimension) => {
    const point = { dimension: String(dimension) };
    METHOD_ORDER.forEach((method) => {
      const record = candidates.find((r) => r.method === method && r.dimension === dimension);
      point[method] = record?.accuracy ?? null;
    });
    return point;
  }), [candidates]);

  const reconstructionChart = useMemo(() => DIMENSIONS.map((dimension) => {
    const point = { dimension: String(dimension) };
    METHOD_ORDER.forEach((method) => {
      const record = candidates.find((r) => r.method === method && r.dimension === dimension);
      point[method] = record?.mse ?? null;
    });
    return point;
  }), [candidates]);

  const filteredRows = useMemo(() => rows.filter((r) => {
    const methodMatch = methodFilter === "All methods" || methodLabel(r.method) === methodFilter;
    const dimensionMatch = dimensionFilter === "All dimensions" || String(r.dimension) === dimensionFilter;
    const searchMatch = `${methodLabel(r.method)} ${r.dimension} ${r.evidence}`.toLowerCase().includes(search.toLowerCase());
    return methodMatch && dimensionMatch && searchMatch;
  }), [rows, methodFilter, dimensionFilter, search]);

  const comparisonRows = candidates.filter((r) =>
    r.dimension === Number(compareDimension) && [compareA, compareB].includes(r.method)
  );
  const currentNav = NAV.find((n) => n.id === activePage);

  const formatKB = (v) => v == null ? "—" : v >= 1000 ? `${(v / 1024).toFixed(2)} MB` : `${pretty(v, 1)} KB`;

  function renderOverview() {
    return <>
      <div className="welcome-row">
        <div>
          <div className="eyebrow"><span className="live-dot" /> PHASE 5 · CONSOLIDATED BENCHMARK</div>
          <h2>Understand the latent space.</h2>
          <p>Explore how compact representations preserve human activity structure under embedded constraints.</p>
        </div>
        <button className="button secondary" onClick={loadData}><RefreshCw size={15} /> Refresh data</button>
      </div>

      <div className="metric-grid">
        <MetricCard label="Benchmark records" value={rows.length || "—"} detail="Including raw sensor baseline" icon={FileBarChart} accent="blue" />
        <MetricCard label="Top 8-subject balanced accuracy" value={bestRecord ? `${pretty(bestRecord.accuracy, 2)}%` : "—"} detail={bestRecord ? `${methodLabel(bestRecord.method)} · d = ${bestRecord.dimension}` : "From benchmark CSV"} icon={Activity} accent="purple" />
        <MetricCard label="Raw input dimensions" value="6,912" detail="256 samples × 27 IMU channels" icon={Database} accent="teal" />
        <MetricCard label="Compression at d = 32" value="216×" detail="128 bytes per window" icon={Zap} accent="amber" />
      </div>

      <div className="content-grid overview-grid">
        <Panel title="Balanced accuracy by latent dimension" subtitle="8-subject macro balanced accuracy · higher is better" className="chart-panel">
          <div className="chart-legend">
            {METHOD_ORDER.map((m) => <span key={m}><i style={{ background: METHOD_META[m].color }} />{methodLabel(m)}</span>)}
          </div>
          <div className="chart-wrap large-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={accuracyChart} margin={{ top: 12, right: 18, left: -14, bottom: 0 }}>
                <CartesianGrid stroke="#263244" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="dimension" stroke="#718096" tickLine={false} axisLine={false} tick={{ fill: "#8492a6", fontSize: 12 }} />
                <YAxis domain={[0, 100]} stroke="#718096" tickLine={false} axisLine={false} tick={{ fill: "#8492a6", fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
                <Tooltip contentStyle={{ background: "#111a29", border: "1px solid #2b3a50", borderRadius: 10, color: "#eef3fb" }} formatter={(v) => v == null ? "Not reported" : `${Number(v).toFixed(2)}%`} labelFormatter={(l) => `Latent dimension: ${l}`} />
                <ReferenceLine y={raw?.accuracy ?? 56.91} stroke="#8391a5" strokeDasharray="5 5" label={{ value: "Raw baseline", fill: "#9aa8ba", fontSize: 10, position: "insideTopRight" }} />
                {METHOD_ORDER.map((m) => <Line key={m} type="monotone" dataKey={m} name={methodLabel(m)} stroke={METHOD_META[m].color} strokeWidth={2.4} connectNulls={false} dot={{ r: 3, strokeWidth: 0 }} activeDot={{ r: 5, stroke: "#0b1220", strokeWidth: 2 }} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-note"><span className="note-dot" /> Dashed line represents the raw 6,912-dimensional baseline.</div>
        </Panel>

        <Panel title="Research highlights" subtitle="Findings reported in Phase 5">
          <div className="highlight-list">
            <div className="highlight-item">
              <span className="highlight-icon violet"><Sparkles size={17} /></span>
              <div><strong>Algorithmic MVD · PCA-32</strong><p>70.65% balanced accuracy at 216× compression, with 0.9455 trustworthiness.</p></div>
            </div>
            <div className="highlight-item">
              <span className="highlight-icon teal"><Cpu size={17} /></span>
              <div><strong>Ultra-constrained · DSC-AE-32 INT8</strong><p>66.25% balanced accuracy with 10 KB Flash and 10.8 KB peak RAM, as reported in the deployment summary.</p></div>
            </div>
            <div className="highlight-item">
              <span className="highlight-icon amber"><ArrowDownRight size={17} /></span>
              <div><strong>Critical dimension boundary</strong><p>The reported transition region is between 8 and 16 latent dimensions.</p></div>
            </div>
          </div>
          <div className="callout">
            <ShieldCheck size={17} />
            <span>Primary benchmark metric excludes subject 109 because its test fold contains only one activity class.</span>
          </div>
        </Panel>
      </div>

      <Panel title="Representation families" subtitle="Five approaches evaluated in the benchmark">
        <div className="family-grid">
          {METHOD_ORDER.map((m, i) => (
            <button className="family-card" key={m} onClick={() => { setMethodFilter(methodLabel(m)); setActivePage("models"); }}>
              <span className="family-index">0{i + 1}</span>
              <span className="family-swatch" style={{ background: METHOD_META[m].color }} />
              <strong>{methodLabel(m)}</strong>
              <small>{METHOD_META[m].family}</small>
              <span className="family-link">Explore model <ArrowUpRight size={13} /></span>
            </button>
          ))}
        </div>
      </Panel>
    </>;
  }

  function renderModels() {
    return <>
      <div className="page-intro"><div><div className="eyebrow">BENCHMARK EXPLORER</div><h2>Representation models</h2><p>Inspect each method and latent dimension using the consolidated Phase 5 results.</p></div></div>
      <Panel title="Benchmark records" subtitle={`${filteredRows.length} records match your filters`}>
        <div className="table-toolbar">
          <label className="search-box"><Search size={15} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search method or evidence..." /></label>
          <select value={methodFilter} onChange={(e) => setMethodFilter(e.target.value)}><option>All methods</option>{METHOD_ORDER.map((m) => <option key={m}>{methodLabel(m)}</option>)}</select>
          <select value={dimensionFilter} onChange={(e) => setDimensionFilter(e.target.value)}><option>All dimensions</option>{[6912, ...DIMENSIONS].map((d) => <option key={d} value={d}>{d.toLocaleString()}</option>)}</select>
          <button className="button subtle" onClick={() => { setSearch(""); setMethodFilter("All methods"); setDimensionFilter("All dimensions"); }}>Reset</button>
        </div>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Method</th><th>Latent d</th><th>Bal. accuracy</th><th>Macro-F1</th><th>Compression</th><th>Trustworthiness</th><th>Recon. MSE</th><th>Flash</th><th>Latency</th></tr></thead>
            <tbody>
              {filteredRows.map((r, i) => <tr key={`${r.method}-${r.dimension}-${i}`}>
                <td><span className="method-cell"><i style={{ background: METHOD_META[r.method]?.color || "#7f8da3" }} />{methodLabel(r.method)}</span></td>
                <td className="mono">{r.dimension?.toLocaleString() ?? "—"}</td>
                <td className="strong-cell">{r.accuracy == null ? "—" : `${pretty(r.accuracy)}%`}</td>
                <td>{r.macroF1 == null ? "—" : `${pretty(r.macroF1)}%`}</td>
                <td>{r.compression == null ? "—" : `${pretty(r.compression, 0)}×`}</td>
                <td>{pretty(r.trustworthiness, 4)}</td>
                <td>{pretty(r.mse, 4)}</td>
                <td>{formatKB(r.flash)}</td>
                <td>{r.latency == null ? "—" : `${pretty(r.latency, 3)} ms`}</td>
              </tr>)}
            </tbody>
          </table>
          {!filteredRows.length && <EmptyState message="No records match these filters." />}
        </div>
      </Panel>
    </>;
  }

  function renderPerformance() {
    return <>
      <div className="page-intro"><div><div className="eyebrow">METRIC ANALYSIS</div><h2>Performance analysis</h2><p>Compare activity separability with reconstruction quality. These measure different properties of a representation.</p></div></div>
      <div className="content-grid">
        <Panel title="Balanced accuracy vs dimension" subtitle="8-subject macro balanced accuracy (%) · higher is better">
          <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={accuracyChart} margin={{ top: 10, right: 16, left: -12, bottom: 0 }}><CartesianGrid stroke="#263244" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="dimension" stroke="#718096" tickLine={false} axisLine={false} /><YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke="#718096" tickLine={false} axisLine={false} /><Tooltip contentStyle={{ background: "#111a29", border: "1px solid #2b3a50", borderRadius: 10 }} formatter={(v) => v == null ? "—" : `${Number(v).toFixed(2)}%`} />{METHOD_ORDER.map((m) => <Line key={m} type="monotone" dataKey={m} name={methodLabel(m)} stroke={METHOD_META[m].color} strokeWidth={2.2} connectNulls dot={{ r: 3 }} />)}</LineChart></ResponsiveContainer></div>
        </Panel>
        <Panel title="Reconstruction MSE vs dimension" subtitle="Mean squared error · lower is better">
          <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={reconstructionChart} margin={{ top: 10, right: 16, left: -12, bottom: 0 }}><CartesianGrid stroke="#263244" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="dimension" stroke="#718096" tickLine={false} axisLine={false} /><YAxis stroke="#718096" tickLine={false} axisLine={false} /><Tooltip contentStyle={{ background: "#111a29", border: "1px solid #2b3a50", borderRadius: 10 }} formatter={(v) => v == null ? "—" : Number(v).toFixed(4)} />{METHOD_ORDER.map((m) => <Line key={m} type="monotone" dataKey={m} name={methodLabel(m)} stroke={METHOD_META[m].color} strokeWidth={2.2} connectNulls dot={{ r: 3 }} />)}</LineChart></ResponsiveContainer></div>
        </Panel>
      </div>
      <div className="insight-strip"><BrainCircuit size={19} /><div><strong>Interpretation matters</strong><p>Reconstruction error decreases as dimension grows, but downstream balanced accuracy does not increase monotonically. A lower reconstruction error is not automatically a better HAR representation.</p></div></div>
    </>;
  }

  function renderEdge() {
    const edgeRows = candidates.filter((r) => r.flash != null && r.flash > 0).sort((a, b) => a.flash - b.flash);
    return <>
      <div className="page-intro"><div><div className="eyebrow">EMBEDDED FEASIBILITY</div><h2>Edge deployment</h2><p>Inspect the reported model footprint and per-window latency. Keep deployment artifacts distinct from benchmark encoder timing.</p></div></div>
      <div className="metric-grid three">
        <MetricCard label="PCA-32 · algorithmic MVD" value={pcaWinner ? `${pcaWinner.flash_size_kb} KB` : "891 KB"} detail="Reported Flash footprint" icon={Cpu} accent="blue" />
        <MetricCard label="DSC-AE-32 · INT8" value={neuralWinner ? `${neuralWinner.flash_size_kb} KB` : "10 KB"} detail="Reported Flash footprint" icon={Zap} accent="teal" />
        <MetricCard label="INT8 peak RAM" value={neuralWinner ? `${neuralWinner.peak_ram_kb} KB` : "10.8 KB"} detail="Deployment summary value" icon={Gauge} accent="purple" />
      </div>
      <div className="content-grid">
        <Panel title="Model Flash footprint" subtitle="Phase 5 CSV · KB · logarithmic scale">
          <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={edgeRows} layout="vertical" margin={{ top: 6, right: 25, left: 12, bottom: 6 }}><CartesianGrid stroke="#263244" strokeDasharray="3 5" horizontal={false} /><XAxis type="number" scale="log" domain={[10, 20000]} stroke="#718096" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} /><YAxis type="category" dataKey={(d) => `${methodLabel(d.method)} · ${d.dimension}`} width={105} stroke="#718096" tickLine={false} axisLine={false} tick={{ fontSize: 10, fill: "#9eacc0" }} /><Tooltip contentStyle={{ background: "#111a29", border: "1px solid #2b3a50", borderRadius: 10 }} formatter={(v) => `${Number(v).toLocaleString()} KB`} /><Bar dataKey="flash" name="Flash" fill="#8b9cff" radius={[0, 5, 5, 0]} barSize={13} /></BarChart></ResponsiveContainer></div>
        </Panel>
        <Panel title="Deployment candidates" subtitle="Distinct deployment claims from the Phase 5 summary">
          <div className="deployment-card">
            <div className="deployment-top"><span className="pill purple">ALGORITHMIC MVD</span><span className="deployment-dim">d = 32</span></div>
            <h3>PCA-32 · C99 inference</h3>
            <div className="deployment-stats"><div><b>{pcaWinner?.flash_size_kb ?? 891} KB</b><small>Flash</small></div><div><b>{pcaWinner?.inference_latency_ms ?? 0.207} ms</b><small>C99 inference</small></div><div><b>{pretty((pcaWinner?.eight_subject_balanced_accuracy ?? 0.7065) * 100, 2)}%</b><small>Balanced accuracy</small></div></div>
            <p>Reported target: microcontrollers with at least 1 MB Flash.</p>
          </div>
          <div className="deployment-card">
            <div className="deployment-top"><span className="pill teal">ULTRA-CONSTRAINED</span><span className="deployment-dim">d = 32</span></div>
            <h3>DSC-AE-32 · INT8</h3>
            <div className="deployment-stats"><div><b>{neuralWinner?.flash_size_kb ?? 10} KB</b><small>Flash</small></div><div><b>{neuralWinner?.peak_ram_kb ?? 10.8} KB</b><small>Peak RAM</small></div><div><b>{pretty((neuralWinner?.eight_subject_balanced_accuracy ?? 0.6625) * 100, 2)}%</b><small>Balanced accuracy</small></div></div>
            <p>Reported target: deeply constrained MCUs with ≤32 KB Flash and ≤16 KB RAM.</p>
          </div>
        </Panel>
      </div>
      <div className="callout warning"><CircleHelp size={17} /><span>The CSV's DSC-AE-32 row is the FP32 Phase 4 record (about 40 KB Flash and 0.0657 ms/win). The INT8 figures above come from the separate deployment summary. PCA's 0.207 ms C99 inference figure is also distinct from the CSV encoder timing. Do not compare these timing fields as if they were measured in the same execution path.</span></div>
    </>;
  }

  function renderCompare() {
    const rowA = comparisonRows.find((r) => r.method === compareA);
    const rowB = comparisonRows.find((r) => r.method === compareB);
    const compareMetric = (label, key, suffix = "", digits = 2) => (
      <div className="compare-row" key={key}><span>{label}</span><strong>{rowA?.[key] == null ? "—" : `${pretty(rowA[key], digits)}${suffix}`}</strong><strong>{rowB?.[key] == null ? "—" : `${pretty(rowB[key], digits)}${suffix}`}</strong></div>
    );
    return <>
      <div className="page-intro"><div><div className="eyebrow">SIDE-BY-SIDE EVALUATION</div><h2>Compare representations</h2><p>Choose two methods at the same latent dimension. Missing combinations remain unavailable rather than being interpolated.</p></div></div>
      <Panel title="Configure comparison" subtitle="Select methods and a shared latent dimension">
        <div className="compare-controls">
          <label>Method A<select value={compareA} onChange={(e) => setCompareA(e.target.value)}>{METHOD_ORDER.map((m) => <option key={m} value={m}>{methodLabel(m)}</option>)}</select></label>
          <label>Method B<select value={compareB} onChange={(e) => setCompareB(e.target.value)}>{METHOD_ORDER.map((m) => <option key={m} value={m}>{methodLabel(m)}</option>)}</select></label>
          <label>Latent dimension<select value={compareDimension} onChange={(e) => setCompareDimension(e.target.value)}>{DIMENSIONS.map((d) => <option key={d} value={d}>{d}</option>)}</select></label>
        </div>
        <div className="compare-table">
          <div className="compare-head"><span>Metric</span><strong>{methodLabel(compareA)}</strong><strong>{methodLabel(compareB)}</strong></div>
          {compareMetric("8-sub balanced accuracy", "accuracy", "%")}
          {compareMetric("Macro-F1", "macroF1", "%")}
          {compareMetric("Compression ratio", "compression", "×", 0)}
          {compareMetric("Trustworthiness", "trustworthiness", "", 4)}
          {compareMetric("Reconstruction MSE", "mse", "", 4)}
          {compareMetric("Model Flash", "flash", " KB", 1)}
          {compareMetric("CSV latency", "latency", " ms", 3)}
        </div>
        {(!rowA || !rowB) && <div className="callout"><CircleHelp size={16} /><span>{!rowA && !rowB ? "Neither method has a record at this dimension." : `The selected dimension is not reported for ${!rowA ? methodLabel(compareA) : methodLabel(compareB)}.`} Try another dimension.</span></div>}
      </Panel>
    </>;
  }

  function renderDataset() {
    const facts = [
      ["Sensor placements", "Hand · Chest · Ankle", "3 IMUs"],
      ["Primary channels", "9 per IMU", "27 channels"],
      ["Sampling frequency", "100 Hz", "Per sensor stream"],
      ["Window length", "256 samples", "2.56 seconds"],
      ["Window stride", "128 samples", "50% overlap"],
      ["Flattened input", "6,912 features", "256 × 27"],
      ["Evaluation windows", "14,973", "Pure non-transient windows"],
      ["Activity classes", "12", "Dominant activity ≠ 0"],
      ["Cross-validation", "9 folds", "LOSO · subjects 101–109"]
    ];
    return <>
      <div className="page-intro"><div><div className="eyebrow">DATA CONTRACT</div><h2>Dataset & protocol</h2><p>Sensor layout, windowing and evaluation protocol documented by the MotionLatent project.</p></div></div>
      <div className="dataset-grid">
        {facts.map(([label, value, note]) => <div className="dataset-card" key={label}><span>{label}</span><strong>{value}</strong><small>{note}</small></div>)}
      </div>
      <Panel title="Sensor channel groups" subtitle="27 primary channels across three body locations">
        <div className="sensor-grid">
          {["Hand", "Chest", "Ankle"].map((place, i) => <div className="sensor-card" key={place}><div className={`sensor-number sensor-${i}`}>0{i + 1}</div><h3>{place} IMU</h3><p>Accelerometer · X / Y / Z</p><p>Gyroscope · X / Y / Z</p><p>Magnetometer · X / Y / Z</p><span>9 channels</span></div>)}
        </div>
      </Panel>
      <div className="callout"><ShieldCheck size={17} /><span>LOSO protocol: scalers and representation transforms are fitted on the eight training subjects only. Subject 109 is retained in the reported nine-fold results, but excluded from the primary eight-subject macro metric because its test fold contains a single activity class.</span></div>
    </>;
  }

  function renderContent() {
    if (loading) return <div className="loading-state"><span className="spinner" /><strong>Loading benchmark artifacts</strong><p>Reading the Phase 5 CSV and summary JSON…</p></div>;
    if (error) return <div className="error-state"><CircleHelp size={24} /><h3>Could not load benchmark data</h3><p>{error}</p><button className="button primary" onClick={loadData}><RefreshCw size={15} /> Try again</button></div>;
    switch (activePage) {
      case "models": return renderModels();
      case "performance": return renderPerformance();
      case "edge": return renderEdge();
      case "compare": return renderCompare();
      case "dataset": return renderDataset();
      default: return renderOverview();
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Activity size={21} /></div>
          <div><strong>Motion<span>Latent</span></strong><small>RESEARCH PLATFORM</small></div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav>
          {NAV.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} className={`nav-item ${activePage === item.id ? "active" : ""}`} onClick={() => { setActivePage(item.id); setMobileNav(false); }}>
              <Icon size={17} /><span>{item.label}</span>{item.id === "overview" && <i className="nav-active-dot" />}
            </button>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="protocol-mini"><div className="protocol-icon"><ShieldCheck size={16} /></div><div><strong>Validated protocol</strong><small>Phase 1–5 artifacts</small></div><CheckCircle2 size={15} className="protocol-check" /></div>
          <a href="https://github.com/Shivam16bh/MotionLatent" target="_blank" rel="noreferrer" className="repo-link">View GitHub repository <ArrowUpRight size={13} /></a>
          <div className="sidebar-version">MOTIONLATENT · PHASE 5</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileNav(!mobileNav)}><Menu size={19} /></button>
          <div className="breadcrumb">MotionLatent <span>/</span> <strong>{currentNav?.label || "Overview"}</strong></div>
          <div className="topbar-right"><span className="data-status"><i /> DATA CONNECTED</span><span className="topbar-divider" /><span className="phase-label">PHASE 5 BENCHMARK</span></div>
        </header>
        <div className="page-content">
          {renderContent()}
          <footer className="footer"><span>MotionLatent · Research benchmark dashboard</span><span>Data source: <code>reports/phase5_comparison.csv</code> + <code>phase5_summary.json</code></span></footer>
        </div>
      </main>
      {mobileNav && <button className="mobile-overlay" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
    </div>
  );
}

export default App;
