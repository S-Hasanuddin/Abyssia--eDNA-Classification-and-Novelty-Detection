import { Dna } from "lucide-react";

const FooterSection = () => {
  return (
    <footer className="py-10 border-t border-border bg-background">
      <div className="container mx-auto px-6 text-center">
        <div className="flex items-center justify-center gap-2 text-primary mb-3">
          <Dna className="w-4 h-4" />
          <span className="font-bold">Abyssia</span>
        </div>
        <p className="text-sm text-muted-foreground">
          Identifying Taxonomy and Assessing Deep Sea Biodiversity Using Self-Supervised AI Pipeline
        </p>
        <p className="text-xs text-muted-foreground mt-2">
          Final Year Project · Dept. of CSE, Methodist College of Engineering & Technology, Hyderabad · 2025–2026
        </p>
      </div>
    </footer>
  );
};

export default FooterSection;