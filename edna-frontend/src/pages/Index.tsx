import { useState } from "react";
import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import StatsSection from "@/components/StatsSection";
import UploadSection from "@/components/UploadSection";
import PipelineSection from "@/components/PipelineSection";
import ResultsSection from "@/components/ResultsSection";
import AboutSection from "@/components/AboutSection";
import FooterSection from "@/components/FooterSection";
import CursorGlow from "@/components/CursorGlow";

const Index = () => {
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);

  return (
    <div className="min-h-screen bg-background">
      <CursorGlow />
      <Navbar />
      <HeroSection />
      <StatsSection />
      <UploadSection onResult={setAnalysisResult} />
      <PipelineSection />
      <ResultsSection result={analysisResult} />
      <AboutSection />
      <FooterSection />
    </div>
  );
};

export default Index;
