import { motion } from "framer-motion";
import dnaImage from "@/assets/dna-glow.png";

const AboutSection = () => {
  return (
    <section id="about" className="py-24 gradient-ocean">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16 items-center max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              About <span className="text-primary glow-text">This Project</span>
            </h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              This system was developed as a final year project at the Department of Computer Science and Engineering,
              Methodist College of Engineering & Technology (MCET), Hyderabad, for the academic year 2025–2026.
              It addresses the critical limitations of traditional eDNA analysis pipelines in deep-sea biodiversity assessment.
            </p>
            <p className="text-muted-foreground leading-relaxed mb-6">
              By combining a BERT-style transformer encoder (96M parameters) with 6-mer tokenization, self-supervised
              representation learning, and HDBSCAN-based unsupervised novelty detection, our pipeline achieves superior
              multi-rank taxonomic classification while discovering previously unknown marine species — all deployed via
              a FastAPI backend with a React web frontend.
            </p>

            <div className="mb-6">
              <h3 className="text-sm font-semibold text-primary mb-3">Built by</h3>
              <div className="flex flex-wrap gap-3">
                {[
                  "S M Hasanuddin",
                  "Amer Ali Khan",
                  "Aryan Asthana",
                ].map((name) => (
                  <span key={name} className="px-4 py-2 rounded-lg bg-secondary border border-primary/20 text-foreground font-medium text-sm">
                    {name}
                  </span>
                ))}
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-sm font-semibold text-primary mb-3">Under the Guidance of</h3>
              <span className="px-4 py-2 rounded-lg bg-secondary border border-primary/20 text-foreground font-medium text-sm">
                Mrs. J. Harika — Assistant Professor, Dept. of CSE
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {[
                { label: "Self-Supervised Learning", desc: "Reference-free sequence embeddings" },
                { label: "Transformer Encoder", desc: "96M param BERT-style architecture" },
                { label: "Novelty Detection", desc: "HDBSCAN clustering for unknown taxa" },
                { label: "Interpretable Output", desc: "Prototype-based explanations" },
              ].map((item) => (
                <div key={item.label} className="p-3 rounded-lg bg-secondary border border-border">
                  <div className="text-sm font-semibold text-foreground">{item.label}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">{item.desc}</div>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="flex justify-center"
          >
            <img src={dnaImage} alt="DNA helix visualization" className="w-80 h-80 object-contain animate-float drop-shadow-[0_0_30px_hsl(175,80%,45%,0.3)]" />
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;
