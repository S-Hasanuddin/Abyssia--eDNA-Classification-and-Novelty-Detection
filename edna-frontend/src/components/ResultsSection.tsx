import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";
import { Download } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// ── Demo data (shown when no real result is present) ─────────────────────────

const baseTaxonomy = [
  { name: "Cnidaria",    base: 28, color: "hsl(175, 80%, 45%)" },
  { name: "Arthropoda",  base: 22, color: "hsl(185, 90%, 50%)" },
  { name: "Annelida",    base: 15, color: "hsl(210, 70%, 50%)" },
  { name: "Chordata",    base: 12, color: "hsl(195, 85%, 40%)" },
  { name: "Novel Taxa",  base: 18, color: "hsl(45,  90%, 55%)" },
  { name: "Other",       base:  5, color: "hsl(200, 15%, 40%)" },
];

const baseAccuracy = [
  { rank: "Domain",  base: 99.18 },
  { rank: "Kingdom", base: 90.13 },
  { rank: "Phylum",  base: 84.99 },
  { rank: "Class",   base: 81.55 },
  { rank: "Order",   base: 65.15 },
  { rank: "Family",  base: 48.15 },
];

const demoNovelTaxa = [
  { cluster: "OTU-N01", reads: 1247, confidence: 0.94, taxon: "Unknown Cnidarian" },
  { cluster: "OTU-N02", reads:  834, confidence: 0.87, taxon: "Potential new Annelid" },
  { cluster: "OTU-N03", reads:  562, confidence: 0.91, taxon: "Uncharacterized Protist" },
];

const CHART_COLORS = [
  "hsl(175, 80%, 45%)", "hsl(185, 90%, 50%)", "hsl(210, 70%, 50%)",
  "hsl(195, 85%, 40%)", "hsl(45,  90%, 55%)", "hsl(200, 15%, 40%)",
  "hsl(220, 70%, 55%)", "hsl(160, 70%, 45%)",
];

const jitter = (val: number, range: number) =>
  +(val + (Math.random() - 0.5) * range).toFixed(1);

// ── Types ─────────────────────────────────────────────────────────────────────

type SeqRecord = Record<string, unknown>;
type ApiResult  = Record<string, unknown>;

interface Props {
  result: Record<string, unknown> | null;
}

// ── Derivation helpers ────────────────────────────────────────────────────────

/** Build pie-chart data from the `classified` array + novel candidate count. */
function deriveTaxonomy(result: ApiResult) {
  const classified = result.classified as SeqRecord[] | undefined;
  const summary     = result.summary   as Record<string, number> | undefined;
  if (!classified || !summary) return null;

  const counts: Record<string, number> = {};
  for (const seq of classified) {
    const tax     = seq.taxonomy as Record<string, Record<string, unknown>> | undefined;
    const phylum  = tax?.phylum;
    const name    = phylum?.confident ? (phylum.taxon as string) : "Other";
    counts[name]  = (counts[name] ?? 0) + 1;
  }

  const novelCandidates = summary.novel_candidates ?? 0;
  if (novelCandidates > 0) {
    counts["Novel Taxa"] = novelCandidates;
  }

  const total = summary.total_sequences || 1;
  return Object.entries(counts).map(([name, count], i) => ({
    name,
    value: Math.round((count / total) * 100),
    color: CHART_COLORS[i % CHART_COLORS.length],
  }));
}

/** Compute per-rank confidence rate across all classified sequences. */
function deriveAccuracy(result: ApiResult) {
  const classified = result.classified as SeqRecord[] | undefined;
  if (!classified || classified.length === 0) return null;

  const ranks = ["domain", "kingdom", "phylum", "class", "order", "family"];
  return ranks.map((rank) => {
    const confident = classified.filter((seq) => {
      const tax = seq.taxonomy as Record<string, Record<string, unknown>> | undefined;
      return tax?.[rank]?.confident === true;
    }).length;
    return {
      rank: rank.charAt(0).toUpperCase() + rank.slice(1),
      accuracy: +((confident / classified.length) * 100).toFixed(1),
    };
  });
}

/** Extract novel cluster cards from the `novel_taxa.clusters` dict. */
function deriveNovelTaxa(result: ApiResult) {
  const novelTaxa = result.novel_taxa as Record<string, unknown> | null | undefined;
  if (!novelTaxa) return null;

  const clusters = novelTaxa.clusters as Record<string, Record<string, unknown>> | undefined;
  if (!clusters || Object.keys(clusters).length === 0) return null;

  return Object.entries(clusters).map(([, cluster], i) => ({
    cluster:    `OTU-N${String(i + 1).padStart(2, "0")}`,
    reads:      (cluster.size        as number) ?? 0,
    confidence: (cluster.novelty_score as number) ?? 0,
    taxon:      (cluster.reported_taxon as string) ?? "Unknown",
    assessment: (cluster.assessment  as string) ?? "",
  }));
}

// ── Novelty score badge for single-sequence card ──────────────────────────────
// Shown when the sequence didn't reach the minimum classification rank (order).
// The "novelty score" for single sequences is derived as 1 - max_confidence
// across all ranks, giving a rough measure of how unlike known taxa it is.
function SingleSeqNoveltyBadge({ taxonomy }: {
  taxonomy: Record<string, Record<string, unknown>>;
}) {
  const ranks = ["domain", "kingdom", "phylum", "class", "order", "family"];
  const maxConf = Math.max(
    ...ranks.map(r => (taxonomy[r]?.confidence as number) ?? 0)
  );
  const noveltyScore = +(1 - maxConf).toFixed(3);
  const pct = (noveltyScore * 100).toFixed(1);

  // Colour the badge based on novelty strength
  const color =
    noveltyScore >= 0.7 ? "text-orange-400 border-orange-400/40 bg-orange-400/10" :
    noveltyScore >= 0.5 ? "text-yellow-400 border-yellow-400/40 bg-yellow-400/10" :
                          "text-cyan-400   border-cyan-400/40   bg-cyan-400/10";

  return (
    <div className={`mt-4 p-3 rounded-lg border ${color} flex items-start gap-3`}>
      <span className="text-lg leading-none">🔬</span>
      <div>
        <div className="text-xs font-semibold uppercase tracking-wide mb-0.5">
          Novel Candidate
        </div>
        <div className="text-xs leading-snug opacity-90">
          This sequence did not reach Order-level confidence. It has been
          routed to HDBSCAN novel clustering in batch mode.
        </div>
        <div className="mt-2 flex items-center gap-2">
          <div className="text-xs font-mono font-bold">{pct}% novelty score</div>
          <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
            <div
              className="h-full rounded-full bg-current transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

const ResultsSection = ({ result }: Props) => {
  const isSingleSeq  = result?.single_sequence === true;
  const isBatchResult = result && !isSingleSeq;
  const liveResult   = isBatchResult ? result as ApiResult : null;

  // Chart state — seeded from demo data, overwritten when real result arrives
  const [taxonomyData, setTaxonomyData] = useState(() =>
    baseTaxonomy.map(t => ({ name: t.name, value: t.base, color: t.color }))
  );
  const [accuracyData, setAccuracyData] = useState(() =>
    baseAccuracy.map(d => ({ rank: d.rank, accuracy: d.base }))
  );
  const [novelCards, setNovelCards]     = useState<typeof demoNovelTaxa | null>(null);
  const [tick, setTick]                 = useState(0);

  // Populate charts from real batch result
  useEffect(() => {
    if (!liveResult) return;

    const tax = deriveTaxonomy(liveResult);
    if (tax) setTaxonomyData(tax);

    const acc = deriveAccuracy(liveResult);
    if (acc) setAccuracyData(acc);

    const novel = deriveNovelTaxa(liveResult);
    setNovelCards(novel);   // null = hide section; array = show cards
  }, [liveResult]);

  // Demo jitter — only when no real result
  useEffect(() => {
    if (liveResult) return;
    const id = setInterval(() => {
      setTaxonomyData(baseTaxonomy.map(t => ({
        name:  t.name,
        value: Math.max(1, Math.round(t.base + (Math.random() - 0.5) * 6)),
        color: t.color,
      })));
      setAccuracyData(baseAccuracy.map(d => ({
        rank: d.rank, accuracy: jitter(d.base, 2),
      })));
      setTick(n => n + 1);
    }, 2500);
    return () => clearInterval(id);
  }, [liveResult]);

  // ── Novel taxa section metadata
  const novelMeta = liveResult
    ? (() => {
        const nt = liveResult.novel_taxa as Record<string, unknown> | null | undefined;
        if (!nt) return null;
        const n        = (nt.n_clusters       as number) ?? "—";
        const frac     = (nt.noise_fraction   as number);
        const fracStr  = frac != null ? (frac * 100).toFixed(1) + "% noise ratio" : "—";
        return `${n} stable clusters · ${fracStr}`;
      })()
    : "16 stable clusters · 1.5% noise ratio";

  // ── PDF
  const hasPdf  = liveResult?.has_pdf  === true;
  const jobId   = liveResult?.job_id   as string | undefined;

  // ── Single-sequence result fields
  const singleSeq = isSingleSeq ? result as ApiResult : null;
  const isNovelCandidate = singleSeq?.specific_enough === false;
  const singleTaxonomy = singleSeq?.taxonomy as
    Record<string, Record<string, unknown>> | undefined;

  return (
    <section id="results" className="py-24 gradient-ocean particle-bg">
      <div className="container mx-auto px-6">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            {liveResult ? "Analysis" : isSingleSeq ? "Classification" : "Sample"}{" "}
            <span className="text-primary glow-text">Results</span>
          </h2>
          <p className="text-muted-foreground max-w-lg mx-auto">
            {liveResult
              ? "Taxonomic composition, rank-wise accuracy, and novel species detected from your uploaded sample."
              : isSingleSeq
              ? "Single-sequence taxonomic classification result."
              : "Live dashboard showing taxonomic composition, rank-wise accuracy, and novel species detection."}
          </p>
          <div className="flex items-center justify-center gap-2 mt-3">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary" />
            </span>
            <span className="text-xs text-primary font-mono tracking-wide">
              {liveResult ? "REAL RESULTS" : isSingleSeq ? "CLASSIFICATION RESULT" : "LIVE DATA STREAM · DEMO"}
            </span>
          </div>
        </motion.div>

        {/* ── Single-sequence classification card ── */}
        {isSingleSeq && singleSeq && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl mx-auto mb-16 p-6 rounded-xl gradient-card border border-border glow-border"
          >
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-lg font-semibold text-foreground">
                🧬 Sequence Classification
              </h3>
              {/* Novel candidate pill — shown in header when routing flags it */}
              {isNovelCandidate && (
                <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full
                  bg-orange-400/15 text-orange-400 border border-orange-400/30">
                  NOVEL CANDIDATE
                </span>
              )}
            </div>
            <p className="text-xs text-muted-foreground font-mono mb-5">
              ID: {singleSeq.id as string} · {singleSeq.sequence_length as number} bp
            </p>

            {/* Best call */}
            <div className={`flex items-center gap-4 p-4 rounded-lg border mb-5 ${
              isNovelCandidate
                ? "bg-orange-400/10 border-orange-400/30"
                : "bg-primary/10 border-primary/30"
            }`}>
              <div>
                <div className="text-xs text-muted-foreground font-mono uppercase tracking-wide mb-0.5">
                  Best classification · {singleSeq.reported_rank as string}
                  {isNovelCandidate && (
                    <span className="ml-2 text-orange-400">
                      · below min rank (order)
                    </span>
                  )}
                </div>
                <div className={`text-xl font-bold ${
                  isNovelCandidate ? "text-orange-400" : "text-primary"
                }`}>
                  {singleSeq.reported_taxon as string}
                </div>
                <div className="text-sm text-muted-foreground mt-0.5">
                  Confidence: {(((singleSeq.reported_confidence as number) ?? 0) * 100).toFixed(1)}%
                  {singleSeq.fallback_occurred && (
                    <span className="ml-2 text-yellow-400 text-xs">↓ fallback applied</span>
                  )}
                </div>
              </div>
            </div>

            {/* Per-rank breakdown */}
            <h4 className="text-sm font-medium text-foreground mb-3">Full rank breakdown</h4>
            <div className="space-y-2">
              {Object.entries(singleTaxonomy ?? {}).map(([rank, info]) => {
                const conf   = (info.confidence    as number) ?? 0;
                const taxon  = (info.taxon         as string) ?? "—";
                const isConf = info.confident === true;
                const thresh = (info.threshold_used as number) ?? 0;
                return (
                  <div
                    key={rank}
                    className="flex items-center justify-between text-sm px-3 py-2 rounded-lg bg-secondary"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-muted-foreground font-mono w-16 text-xs uppercase">
                        {rank}
                      </span>
                      <span className={`font-medium ${isConf ? "text-foreground" : "text-muted-foreground"}`}>
                        {taxon}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {/* Show threshold used as a faint tick mark on the bar */}
                      <div className="relative w-20 h-1.5 rounded-full bg-border overflow-visible">
                        <div
                          className="h-full rounded-full bg-primary"
                          style={{ width: `${(conf * 100).toFixed(0)}%` }}
                        />
                        {/* Threshold marker */}
                        <div
                          className="absolute top-1/2 -translate-y-1/2 w-px h-3 bg-yellow-400/60"
                          style={{ left: `${(thresh * 100).toFixed(0)}%` }}
                          title={`Threshold: ${(thresh * 100).toFixed(0)}%`}
                        />
                      </div>
                      <span className={`text-xs font-mono w-10 text-right ${
                        isConf ? "text-primary" : "text-muted-foreground"
                      }`}>
                        {(conf * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Novelty score block — only shown when sequence is a novel candidate */}
            {isNovelCandidate && singleTaxonomy && (
              <SingleSeqNoveltyBadge taxonomy={singleTaxonomy} />
            )}
          </motion.div>
        )}

        {/* ── Batch result charts ── */}
        {!isSingleSeq && (
          <div className="grid lg:grid-cols-2 gap-8 max-w-5xl mx-auto">

            {/* Taxonomy Pie */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="p-6 rounded-xl gradient-card border border-border glow-border"
            >
              <h3 className="text-lg font-semibold text-foreground mb-4">Taxonomic Composition</h3>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={taxonomyData}
                    cx="50%" cy="50%"
                    innerRadius={60} outerRadius={100}
                    paddingAngle={3}
                    dataKey="value"
                    isAnimationActive animationDuration={800} animationEasing="ease-in-out"
                  >
                    {taxonomyData.map((entry, i) => (
                      <Cell key={`${i}-${tick}`} fill={entry.color} stroke="transparent" />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "hsl(210 35% 6%)",
                      border: "1px solid hsl(200 20% 15%)",
                      borderRadius: "8px",
                      color: "hsl(185 30% 90%)",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap gap-3 mt-4 justify-center">
                {taxonomyData.map(t => (
                  <div key={t.name} className="flex items-center gap-1.5 text-xs">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ background: t.color }} />
                    <span className="text-muted-foreground">{t.name} ({t.value}%)</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Rank-wise Accuracy Bar */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="p-6 rounded-xl gradient-card border border-border glow-border"
            >
              <h3 className="text-lg font-semibold text-foreground mb-4">
                Rank-wise Classification Accuracy (%)
              </h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={accuracyData}>
                  <XAxis
                    dataKey="rank"
                    tick={{ fill: "hsl(200, 15%, 55%)", fontSize: 11 }}
                    axisLine={false} tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "hsl(200, 15%, 55%)", fontSize: 12 }}
                    axisLine={false} tickLine={false}
                    domain={[0, 100]}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(210 35% 6%)",
                      border: "1px solid hsl(200 20% 15%)",
                      borderRadius: "8px",
                      color: "hsl(185 30% 90%)",
                    }}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, "Accuracy"]}
                  />
                  <Bar
                    dataKey="accuracy"
                    fill="hsl(175, 80%, 45%)"
                    radius={[6, 6, 0, 0]}
                    isAnimationActive animationDuration={800}
                  />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>

            {/* ── Per-sequence individual results ── */}
            {liveResult && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="lg:col-span-2 p-6 rounded-xl gradient-card border border-border glow-border"
              >
                <h3 className="text-lg font-semibold text-foreground mb-1">
                  📋 Individual Sequence Results
                </h3>
                <p className="text-xs text-muted-foreground font-mono mb-4">
                  {(liveResult.summary as Record<string, number>)?.total_sequences ?? 0} sequences ·{" "}
                  {(liveResult.summary as Record<string, number>)?.classified ?? 0} classified ·{" "}
                  {(liveResult.summary as Record<string, number>)?.novel_candidates ?? 0} novel candidates
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-xs text-muted-foreground font-mono border-b border-border">
                        <th className="text-left pb-2 pr-4">ID</th>
                        <th className="text-left pb-2 pr-4">Length</th>
                        <th className="text-left pb-2 pr-4">Rank</th>
                        <th className="text-left pb-2 pr-4">Taxon</th>
                        <th className="text-right pb-2">Confidence</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/40">
                      {((liveResult.classified as SeqRecord[]) ?? []).map((seq) => (
                        <tr key={seq.id as string} className="hover:bg-secondary/30 transition-colors">
                          <td className="py-2 pr-4 font-mono text-xs text-muted-foreground truncate max-w-[120px]">
                            {seq.id as string}
                          </td>
                          <td className="py-2 pr-4 text-xs text-muted-foreground">
                            {seq.sequence_length as number} bp
                          </td>
                          <td className="py-2 pr-4 text-xs font-mono text-foreground capitalize">
                            {seq.reported_rank as string}
                          </td>
                          <td className="py-2 pr-4 text-xs text-foreground">
                            {seq.reported_taxon as string}
                          </td>
                          <td className="py-2 text-right text-xs font-mono text-primary">
                            {(((seq.reported_confidence as number) ?? 0) * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {/* ── Novel Taxa — only shown when novel taxa were actually detected ── */}
            {(liveResult ? novelCards !== null : true) && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="lg:col-span-2 p-6 rounded-xl gradient-card border border-border glow-border"
              >
                <h3 className="text-lg font-semibold text-foreground mb-1">
                  🔬 Novel Taxa Detected (HDBSCAN Clustering)
                </h3>

                <p className="text-xs text-muted-foreground mb-4 font-mono">
                  {novelMeta}
                </p>

                <div className="grid sm:grid-cols-3 gap-4">
                  {(novelCards ?? demoNovelTaxa).map((item) => (
                    <div
                      key={item.cluster}
                      className="p-4 rounded-lg bg-secondary border border-primary/20"
                    >
                      <div className="font-mono text-primary text-sm font-semibold">
                        {item.cluster}
                      </div>
                      <div className="text-foreground font-medium mt-1">{item.taxon}</div>
                      {"assessment" in item && item.assessment && (
                        <div className="text-xs text-muted-foreground mt-1 leading-snug">
                          {item.assessment}
                        </div>
                      )}
                      <div className="text-xs text-muted-foreground mt-2 font-mono tabular-nums">
                        {item.reads.toLocaleString()} reads ·{" "}
                        {(item.confidence * 100).toFixed(0)}% novelty score
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* ── PDF Download — only shown when report is ready ── */}
            {hasPdf && jobId && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="lg:col-span-2 flex justify-center"
              >
                <a
                  href={`${API_BASE}/api/report/${jobId}/pdf`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-8 py-3 rounded-lg bg-primary text-primary-foreground font-semibold hover:brightness-110 transition-all glow-box"
                >
                  <Download className="w-4 h-4" />
                  Download PDF Report
                </a>
              </motion.div>
            )}

          </div>
        )}

      </div>
    </section>
  );
};

export default ResultsSection;
