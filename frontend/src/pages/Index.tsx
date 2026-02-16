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
    // Default scores for now - AssessmentForm can be enhanced to collect actual scores
    const scores = {
      mathematics: 50,
      science: 50,
      language_arts: 50,
      social_studies: 50,
      logical_reasoning: 50,
      creativity: 50,
      communication: 50,
      leadership: 50,
      social_skills: 50,
    };

    const recs = getRecommendations(
      data.academics,
      data.interests,
      data.skills,
      data.hobbies,
      scores,
      "" // preferred location
    );
    
    recs.then((results) => {
      setResults(results);
      setShowAssessment(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }).catch((error) => {
      console.error("Error getting recommendations:", error);
      setShowAssessment(false);
    });
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
