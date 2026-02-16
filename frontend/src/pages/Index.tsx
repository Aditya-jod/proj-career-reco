import { useState, useRef } from "react";
import HeroSection from "@/components/HeroSection";
import AssessmentForm from "@/components/AssessmentForm";
import CareerResults from "@/components/CareerResults";
import { getRecommendations, type CareerPath } from "@/data/careerData";

const Index = () => {
  const [results, setResults] = useState<CareerPath[] | null>(null);
  const [showAssessment, setShowAssessment] = useState(false);
  const assessmentRef = useRef<HTMLDivElement>(null);

  const handleStart = () => {
    setShowAssessment(true);
    setResults(null);
    setTimeout(() => {
      assessmentRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  };

  const handleSubmit = (data: {
    academics: string;
    interests: string[];
    skills: string[];
    hobbies: string[];
  }) => {
    const recs = getRecommendations(data.academics, data.interests, data.skills, data.hobbies);
    setResults(recs);
    setShowAssessment(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleReset = () => {
    setResults(null);
    setShowAssessment(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-background">
      {!results && <HeroSection onStart={handleStart} />}

      {showAssessment && !results && (
        <div ref={assessmentRef}>
          <AssessmentForm onSubmit={handleSubmit} />
        </div>
      )}

      {results && <CareerResults careers={results} onReset={handleReset} />}

      {/* Footer */}
      <footer className="py-8 text-center text-sm text-muted-foreground border-t border-border">
        <p>Career Path Recommender — Helping students find their future.</p>
      </footer>
    </div>
  );
};

export default Index;
