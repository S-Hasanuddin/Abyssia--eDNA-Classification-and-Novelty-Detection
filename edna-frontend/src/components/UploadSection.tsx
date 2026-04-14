import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText, X, Play, Loader2, CheckCircle2, Dna } from "lucide-react";

type AnalysisState = "idle" | "uploading" | "processing" | "complete" | "error";
type InputMode = "file" | "sequence";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const ACCEPTED_EXTENSIONS = ".fasta,.fa,.fna,.fastq,.fq,.fastq.gz,.fq.gz,.csv,.json,.xls,.xlsx";
const ACCEPTED_DESCRIPTION = ".fasta, .fastq, .csv, .json, .xls, .xlsx supported";

const pipelineSteps = [
  "Validating file format...",
  "Preprocessing & quality control...",
  "6-mer tokenization of sequences...",
  "Generating embeddings via Transformer Encoder (96M params)...",
  "Running multi-rank taxonomic classification...",
  "HDBSCAN novelty detection & clustering...",
  "Computing ecological diversity metrics (Shannon, Simpson)...",
  "Generating biodiversity report with confidence scores...",
];

interface Props {
  onResult: (result: Record<string, unknown>) => void;
}

const UploadSection = ({ onResult }: Props) => {
  const [mode, setMode] = useState<InputMode>("file");
  const [files, setFiles] = useState<File[]>([]);
  const [sequence, setSequence] = useState("");
  const [seqId, setSeqId] = useState("");
  const [state, setState] = useState<AnalysisState>("idle");
  const [currentStep, setCurrentStep] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = Array.from(e.dataTransfer.files);
    setFiles(prev => [...prev, ...dropped]);
  }, []);

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Poll job until complete, advancing step display
  const pollJob = async (jobId: string): Promise<Record<string, unknown>> => {
    const POLL_INTERVAL = 1200;
    while (true) {
      await new Promise(r => setTimeout(r, POLL_INTERVAL));
      const res = await fetch(`${API_BASE}/api/job/${jobId}`);
      if (!res.ok) throw new Error(`Job poll failed: ${res.status}`);
      const job = await res.json();

      if (typeof job.step_index === "number") {
        setCurrentStep(Math.min(job.step_index, pipelineSteps.length - 1));
      }

      if (job.status === "complete") {
        const resultRes = await fetch(`${API_BASE}/api/job/${jobId}/result`);
        if (!resultRes.ok) throw new Error("Failed to fetch results");
        return await resultRes.json();
      }

      if (job.status === "failed") {
        throw new Error(job.error ?? "Analysis job failed");
      }
    }
  };

  const runFileAnalysis = async () => {
    if (files.length === 0) return;
    setErrorMsg("");
    setState("uploading");

    try {
      const formData = new FormData();
      formData.append("file", files[0]);

      const uploadRes = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        const err = await uploadRes.json().catch(() => ({ detail: uploadRes.statusText }));
        throw new Error(err.detail ?? "Upload failed");
      }

      const { job_id } = await uploadRes.json();
      setState("processing");
      const result = await pollJob(job_id);
      onResult(result);
      setState("complete");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setErrorMsg(msg);
      setState("error");
    }
  };

  const runSequenceAnalysis = async () => {
    const seq = sequence.trim();
    if (!seq) return;
    setErrorMsg("");
    setState("processing");
    setCurrentStep(0);

    try {
      // Animate steps while waiting for the synchronous inference call
      const stepTimer = setInterval(() => {
        setCurrentStep(prev => Math.min(prev + 1, pipelineSteps.length - 2));
      }, 900);

      const res = await fetch(`${API_BASE}/api/analyze/sequence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence: seq, id: seqId.trim() || "query" }),
      });

      clearInterval(stepTimer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? "Sequence analysis failed");
      }

      const result = await res.json();
      setCurrentStep(pipelineSteps.length - 1);
      onResult({ single_sequence: true, ...result });
      setState("complete");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setErrorMsg(msg);
      setState("error");
    }
  };

  const reset = () => {
    setState("idle");
    setFiles([]);
    setSequence("");
    setSeqId("");
    setCurrentStep(0);
    setErrorMsg("");
  };

  const isRunning = state === "uploading" || state === "processing";

  return (
    <section id="upload" className="py-24 gradient-ocean particle-bg">
      <div className="container mx-auto px-6 max-w-3xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Upload <span className="text-primary glow-text">eDNA Samples</span>
          </h2>
          <p className="text-muted-foreground max-w-lg mx-auto">
            Drop your sequence files or paste a single sequence — our self-supervised AI pipeline handles the rest,
            from quality control to final biodiversity report.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-border gradient-card p-8 glow-border"
        >
          {state === "idle" && (
            <>
              {/* Mode tabs */}
              <div className="flex rounded-lg bg-secondary p-1 mb-6 gap-1">
                <button
                  onClick={() => setMode("file")}
                  className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm font-medium transition-all ${
                    mode === "file"
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Upload className="w-4 h-4" />
                  File Upload
                </button>
                <button
                  onClick={() => setMode("sequence")}
                  className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm font-medium transition-all ${
                    mode === "sequence"
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Dna className="w-4 h-4" />
                  Single Sequence
                </button>
              </div>

              {mode === "file" ? (
                <>
                  <div
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${
                      dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/40"
                    }`}
                    onClick={() => {
                      const input = document.createElement("input");
                      input.type = "file";
                      input.multiple = true;
                      input.accept = ACCEPTED_EXTENSIONS;
                      input.onchange = (e) => {
                        const target = e.target as HTMLInputElement;
                        if (target.files) setFiles(prev => [...prev, ...Array.from(target.files!)]);
                      };
                      input.click();
                    }}
                  >
                    <Upload className="w-10 h-10 text-primary mx-auto mb-4 animate-pulse-glow" />
                    <p className="text-foreground font-medium mb-1">Drop files here or click to browse</p>
                    <p className="text-sm text-muted-foreground">{ACCEPTED_DESCRIPTION}</p>
                  </div>

                  <AnimatePresence>
                    {files.length > 0 && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        className="mt-6 space-y-2"
                      >
                        {files.map((file, i) => (
                          <div key={i} className="flex items-center justify-between px-4 py-2.5 rounded-lg bg-secondary">
                            <div className="flex items-center gap-3">
                              <FileText className="w-4 h-4 text-primary" />
                              <span className="text-sm font-mono text-foreground">{file.name}</span>
                              <span className="text-xs text-muted-foreground">
                                {(file.size / 1024 / 1024).toFixed(1)} MB
                              </span>
                            </div>
                            <button onClick={() => removeFile(i)} className="text-muted-foreground hover:text-destructive">
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                        <button
                          onClick={runFileAnalysis}
                          className="w-full mt-4 flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-primary text-primary-foreground font-semibold hover:brightness-110 transition-all glow-box"
                        >
                          <Play className="w-4 h-4" />
                          Run Analysis
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </>
              ) : (
                /* Single sequence mode */
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1.5">
                      Sequence ID <span className="text-muted-foreground font-normal">(optional)</span>
                    </label>
                    <input
                      type="text"
                      value={seqId}
                      onChange={e => setSeqId(e.target.value)}
                      placeholder="e.g. sample_001"
                      className="w-full px-3 py-2 rounded-lg bg-secondary border border-border text-foreground text-sm placeholder:text-muted-foreground focus:outline-none focus:border-primary/60 transition-colors"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1.5">
                      DNA Sequence <span className="text-destructive">*</span>
                    </label>
                    <textarea
                      value={sequence}
                      onChange={e => setSequence(e.target.value)}
                      placeholder="Paste your DNA sequence here (ACGT, min 50 bp, max 10,000 bp)..."
                      rows={6}
                      className="w-full px-3 py-2 rounded-lg bg-secondary border border-border text-foreground text-sm font-mono placeholder:text-muted-foreground focus:outline-none focus:border-primary/60 transition-colors resize-y"
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      {sequence.trim().length} bp &nbsp;·&nbsp; Accepted bases: A C G T N and IUPAC ambiguity codes
                    </p>
                  </div>
                  <button
                    onClick={runSequenceAnalysis}
                    disabled={sequence.trim().length < 50}
                    className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-primary text-primary-foreground font-semibold hover:brightness-110 transition-all glow-box disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <Play className="w-4 h-4" />
                    Classify Sequence
                  </button>
                </div>
              )}
            </>
          )}

          {isRunning && (
            <div className="py-8">
              <Loader2 className="w-8 h-8 text-primary mx-auto mb-6 animate-spin" />
              <div className="space-y-3 max-w-md mx-auto">
                {pipelineSteps.map((step, i) => (
                  <div key={i} className="flex items-center gap-3">
                    {i < currentStep ? (
                      <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />
                    ) : i === currentStep ? (
                      <Loader2 className="w-4 h-4 text-primary animate-spin shrink-0" />
                    ) : (
                      <div className="w-4 h-4 rounded-full border border-border shrink-0" />
                    )}
                    <span className={`text-sm font-mono ${i <= currentStep ? "text-foreground" : "text-muted-foreground"}`}>
                      {step}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {state === "complete" && (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="py-8 text-center">
              <CheckCircle2 className="w-12 h-12 text-primary mx-auto mb-4 glow-text" />
              <h3 className="text-xl font-bold text-foreground mb-2">Analysis Complete</h3>
              <p className="text-muted-foreground mb-6">Your biodiversity report is ready below.</p>
              <div className="flex gap-3 justify-center">
                <a href="#results" className="px-6 py-2.5 rounded-lg bg-primary text-primary-foreground font-semibold hover:brightness-110 glow-box">
                  View Results
                </a>
                <button onClick={reset} className="px-6 py-2.5 rounded-lg border border-border text-foreground hover:bg-secondary transition-all">
                  New Analysis
                </button>
              </div>
            </motion.div>
          )}

          {state === "error" && (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="py-8 text-center">
              <X className="w-12 h-12 text-destructive mx-auto mb-4" />
              <h3 className="text-xl font-bold text-foreground mb-2">Analysis Failed</h3>
              <p className="text-sm text-muted-foreground mb-6 font-mono max-w-sm mx-auto break-words">{errorMsg}</p>
              <button onClick={reset} className="px-6 py-2.5 rounded-lg bg-primary text-primary-foreground font-semibold hover:brightness-110 glow-box">
                Try Again
              </button>
            </motion.div>
          )}
        </motion.div>
      </div>
    </section>
  );
};

export default UploadSection;
