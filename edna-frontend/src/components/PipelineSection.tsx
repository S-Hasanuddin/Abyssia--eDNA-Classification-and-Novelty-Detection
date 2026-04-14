import { motion } from "framer-motion";
import { Upload, Dna, Brain, Search, BarChart3, Shield } from "lucide-react";

const steps = [
  { icon: Upload, title: "Preprocessing & QC", desc: "Adapter trimming, error correction, fragmentation-aware filtering & contamination screening" },
  { icon: Dna, title: "6-mer Tokenization", desc: "Encoding raw nucleotide sequences into k-mer tokens for transformer input" },
  { icon: Brain, title: "Transformer Encoder", desc: "96M-parameter BERT-style self-supervised embedding generation via masked modelling" },
  { icon: Search, title: "Multi-Rank Classification", desc: "Cascaded taxonomy prediction at Domain, Kingdom, Phylum, Class, Order & Family ranks" },
  { icon: BarChart3, title: "Novelty Detection", desc: "HDBSCAN unsupervised clustering to discover novel OTUs beyond known references" },
  { icon: Shield, title: "Ecological Analysis & Report", desc: "Shannon, Simpson, richness & evenness indices with confidence-calibrated biodiversity report" },
];

const PipelineSection = () => {
  return (
    <section id="pipeline" className="py-24 gradient-ocean relative overflow-hidden">
      {/* Animated background pulse */}
      <motion.div
        className="absolute inset-0 pointer-events-none"
        style={{ background: "radial-gradient(ellipse at 50% 50%, hsla(175, 80%, 45%, 0.03), transparent 70%)" }}
        animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      />

      <div className="container mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            The <span className="text-primary glow-text">Pipeline</span>
          </h2>
          <p className="text-muted-foreground max-w-lg mx-auto">
            End-to-end modular workflow from raw eDNA reads to actionable biodiversity insights — no reference database required.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
          {steps.map((step, i) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, y: 30, rotateX: 15 }}
              whileInView={{ opacity: 1, y: 0, rotateX: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12, type: "spring", stiffness: 80, damping: 15 }}
              whileHover={{
                y: -8,
                scale: 1.02,
                boxShadow: "0 0 40px hsla(175, 80%, 45%, 0.15)",
                borderColor: "hsl(175 80% 45% / 0.5)",
              }}
              className="relative p-6 rounded-xl gradient-card border border-border glow-border group transition-colors cursor-default"
            >
              {/* Step number with pulse */}
              <motion.div
                className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-bold"
                whileInView={{ scale: [0, 1.3, 1] }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.12 + 0.3, duration: 0.5 }}
              >
                {i + 1}
              </motion.div>

              {/* Animated icon */}
              <motion.div
                whileHover={{ rotate: [0, -10, 10, 0], scale: 1.2 }}
                transition={{ duration: 0.5 }}
              >
                <step.icon className="w-8 h-8 text-primary mb-4" />
              </motion.div>

              <h3 className="text-lg font-semibold text-foreground mb-2">{step.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{step.desc}</p>

              {/* Connecting line to next card */}
              {i < steps.length - 1 && i % 3 !== 2 && (
                <motion.div
                  className="hidden lg:block absolute top-1/2 -right-3 w-6 h-px bg-primary/30"
                  initial={{ scaleX: 0 }}
                  whileInView={{ scaleX: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.12 + 0.5 }}
                />
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default PipelineSection;
