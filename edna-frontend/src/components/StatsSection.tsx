import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";

const stats = [
  { value: 99.18, suffix: "%", label: "Domain Accuracy", desc: "Highest rank classification" },
  { value: 84.99, suffix: "%", label: "Phylum Accuracy", desc: "Mid-rank taxonomic precision" },
  { value: 1.24, suffix: "M", label: "Training Sequences", desc: "From 6 public repositories" },
  { value: 0.984, suffix: "", label: "Silhouette Score", desc: "16 HDBSCAN clusters, 1.5% noise" },
];

const AnimatedCounter = ({ value, suffix, duration = 2 }: { value: number; suffix: string; duration?: number }) => {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    let start = 0;
    const end = value;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / (duration * 1000), 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * eased;
      setCount(current);
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [inView, value, duration]);

  const formatted = value >= 1 ? count.toFixed(value % 1 === 0 ? 0 : 2) : count.toFixed(3);

  return (
    <div ref={ref} className="text-3xl md:text-4xl font-bold text-primary glow-text mb-2 tabular-nums">
      {formatted}{suffix}
    </div>
  );
};

const StatsSection = () => {
  return (
    <section className="py-20 gradient-ocean">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 30, scale: 0.9 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.15, type: "spring", stiffness: 100 }}
              whileHover={{ y: -5, boxShadow: "0 0 30px hsla(175, 80%, 45%, 0.15)" }}
              className="text-center p-6 rounded-xl gradient-card border border-border glow-border cursor-default"
            >
              <AnimatedCounter value={stat.value} suffix={stat.suffix} />
              <div className="text-sm font-semibold text-foreground mb-1">{stat.label}</div>
              <div className="text-xs text-muted-foreground">{stat.desc}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default StatsSection;
