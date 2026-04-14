import { motion } from "framer-motion";
import { ArrowDown } from "lucide-react";
import heroImage from "@/assets/hero-deep-sea.jpg";

const FloatingOrb = ({ delay, duration, x, y, size, color }: { delay: number; duration: number; x: string; y: string; size: number; color: string }) => (
  <motion.div
    className="absolute rounded-full blur-xl pointer-events-none"
    style={{ left: x, top: y, width: size, height: size, background: color }}
    animate={{
      y: [0, -30, 10, -20, 0],
      x: [0, 15, -10, 20, 0],
      scale: [1, 1.2, 0.9, 1.1, 1],
      opacity: [0.3, 0.6, 0.3, 0.5, 0.3],
    }}
    transition={{ duration, delay, repeat: Infinity, ease: "easeInOut" }}
  />
);

const DNAHelix = ({ delay }: { delay: number }) => (
  <motion.div
    className="absolute w-px h-32 opacity-20"
    style={{
      left: `${15 + Math.random() * 70}%`,
      top: `${10 + Math.random() * 60}%`,
      background: "linear-gradient(180deg, transparent, hsl(175 80% 45%), transparent)",
    }}
    animate={{ y: [0, -60, 0], opacity: [0, 0.3, 0], scaleY: [0.5, 1, 0.5] }}
    transition={{ duration: 5 + Math.random() * 3, delay, repeat: Infinity, ease: "easeInOut" }}
  />
);

const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background image */}
      <div className="absolute inset-0">
        <img src={heroImage} alt="Deep sea bioluminescent scene" className="w-full h-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-b from-background/80 via-background/50 to-background" />
      </div>

      {/* Animated floating orbs */}
      <FloatingOrb delay={0} duration={8} x="10%" y="20%" size={120} color="hsla(175, 80%, 45%, 0.08)" />
      <FloatingOrb delay={1.5} duration={10} x="75%" y="15%" size={180} color="hsla(195, 85%, 40%, 0.06)" />
      <FloatingOrb delay={3} duration={7} x="60%" y="65%" size={100} color="hsla(185, 90%, 50%, 0.07)" />
      <FloatingOrb delay={2} duration={9} x="25%" y="70%" size={140} color="hsla(210, 70%, 50%, 0.05)" />
      <FloatingOrb delay={4} duration={11} x="85%" y="55%" size={90} color="hsla(175, 80%, 45%, 0.06)" />

      {/* Animated vertical DNA-like lines */}
      {[0, 1.2, 2.5, 3.8, 5].map((d, i) => (
        <DNAHelix key={i} delay={d} />
      ))}

      {/* Particle overlay */}
      <div className="absolute inset-0 particle-bg" />

      {/* Scanning line effect */}
      <motion.div
        className="absolute left-0 right-0 h-px pointer-events-none"
        style={{ background: "linear-gradient(90deg, transparent, hsl(175 80% 45% / 0.3), transparent)" }}
        animate={{ top: ["0%", "100%"] }}
        transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
      />

      <div className="relative z-10 container mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <span className="inline-block px-4 py-1.5 rounded-full border border-primary/30 text-primary text-sm font-mono mb-6 glow-border">
            <motion.span
              animate={{ opacity: [0.7, 1, 0.7] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              Self-Supervised AI Pipeline
            </motion.span>
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.15 }}
          className="text-5xl md:text-7xl font-bold tracking-tight mb-6 glow-text"
        >
          Identifying Taxonomy &
          <br />
          <motion.span
            className="text-primary inline-block"
            animate={{ textShadow: [
              "0 0 20px hsla(175, 80%, 45%, 0.5)",
              "0 0 40px hsla(175, 80%, 45%, 0.8)",
              "0 0 20px hsla(175, 80%, 45%, 0.5)",
            ]}}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          >
            Deep-Sea Biodiversity
          </motion.span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed"
        >
          Upload eDNA samples. Our hybrid self-supervised deep-learning pipeline with a 96M-parameter
          transformer encoder classifies taxonomy at six ranks, detects novel taxa via HDBSCAN clustering,
          and generates comprehensive biodiversity reports.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.45 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <motion.a
            href="#upload"
            className="inline-flex items-center justify-center px-8 py-3.5 rounded-lg bg-primary text-primary-foreground font-semibold transition-all glow-box"
            whileHover={{ scale: 1.05, boxShadow: "0 0 30px hsla(175, 80%, 45%, 0.4)" }}
            whileTap={{ scale: 0.97 }}
          >
            Start Analysis
          </motion.a>
          <motion.a
            href="#pipeline"
            className="inline-flex items-center justify-center px-8 py-3.5 rounded-lg border border-primary/30 text-foreground font-semibold transition-all glow-border"
            whileHover={{ scale: 1.05, borderColor: "hsl(175 80% 45%)" }}
            whileTap={{ scale: 0.97 }}
          >
            View Pipeline
          </motion.a>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 1 }}
          className="mt-16"
        >
          <ArrowDown className="w-5 h-5 text-primary animate-bounce mx-auto" />
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;
