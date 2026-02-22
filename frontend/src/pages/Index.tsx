import { useState, useRef } from "react";
import HeroSection from "@/components/HeroSection";
import AssessmentForm from "@/components/AssessmentForm";
import CareerResults from "@/components/CareerResults";
import { getRecommendations, type CareerPath, type University, type Job } from "@/data/careerData";

interface AppResults {
  careers: CareerPath[];
  universities: University[];
  jobs: Job[];
}

const Index = () => {
  const [results, setResults] = useState<AppResults | null>(null);
  const [showAssessment, setShowAssessment] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
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
    scores: {
      mathematics: number;
      science: number;
      language_arts: number;
      social_studies: number;
      logical_reasoning: number;
      creativity: number;
      communication: number;
      leadership: number;
      social_skills: number;
    };
    preferredLocation: string;
  }) => {
    setIsLoading(true);

    const recs = getRecommendations(
      data.academics,
      data.interests,
      data.skills,
      data.hobbies,
      data.scores,
      data.preferredLocation
    );

    recs
      .then((response) => {
        setResults(response);
        setShowAssessment(false);
        setIsLoading(false);
        window.scrollTo({ top: 0, behavior: "smooth" });
      })
      .catch((error) => {
        console.error("Error getting recommendations:", error);
        setShowAssessment(false);
        setIsLoading(false);
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
          {isLoading ? (
            <section className="py-20 px-4 text-center">
              <div className="max-w-md mx-auto">
                <div className="inline-block w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4" />
                <p className="text-lg font-medium text-foreground">Analysing your profile…</p>
                <p className="text-sm text-muted-foreground mt-1">Our AI is finding your best career matches</p>
              </div>
            </section>
          ) : (
            <AssessmentForm onSubmit={handleSubmit} />
          )}
        </div>
      )}

      {results && (
        <CareerResults
          careers={results.careers}
          universities={results.universities}
          jobs={results.jobs}
          onReset={handleReset}
        />
      )}

      {/* Footer */}
      <footer className="py-8 text-center text-sm text-muted-foreground border-t border-border">
        <p>Career Path Recommender — Helping students find their future.</p>
      </footer>
    </div>
  );
};

export default Index;
