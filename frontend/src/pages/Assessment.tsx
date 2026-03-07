import { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { LogOut, ArrowLeft } from "lucide-react";
import AssessmentForm from "@/components/AssessmentForm";
import CareerResults from "@/components/CareerResults";
import { getRecommendations, type CareerPath, type University, type Job } from "@/data/careerData";
import { useAuth } from "@/context/AuthContext";

interface AppResults {
  careers: CareerPath[];
  universities: University[];
  jobs: Job[];
}

type SubmitData = {
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
};

const Assessment = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();

  const [results, setResults] = useState<AppResults | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Auth guard — redirect unauthenticated visitors to login
  useEffect(() => {
    if (!isAuthenticated) {
      navigate("/login", { state: { from: "/assessment" }, replace: true });
    }
  }, [isAuthenticated, navigate]);

  if (!isAuthenticated) return null;

  const handleSubmit = (data: SubmitData) => {
    setIsLoading(true);
    setApiError(null);

    getRecommendations(
      data.academics,
      data.interests,
      data.skills,
      data.hobbies,
      data.scores,
      data.preferredLocation,
    )
      .then((response) => {
        setResults(response);
        setIsLoading(false);
        window.scrollTo({ top: 0, behavior: "smooth" });
      })
      .catch((error: unknown) => {
        const message =
          error instanceof Error ? error.message : "An unexpected error occurred.";

        // If the token expired / was invalid, log the user out so the auth
        // guard redirects them to the login page automatically.
        if (message.includes("401")) {
          logout();
          return;
        }

        setApiError(`Could not retrieve recommendations: ${message}`);
        setIsLoading(false);
      });
  };

  const handleReset = () => {
    setResults(null);
    setApiError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <nav className="sticky top-0 z-30 flex items-center justify-between px-6 py-3 border-b border-border bg-background/80 backdrop-blur-md">
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Home
        </button>

        <span className="font-display font-bold text-lg text-foreground absolute left-1/2 -translate-x-1/2">
          CareerPath
        </span>

        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground hidden sm:inline">
            Hi, <span className="font-medium text-foreground">{user?.name}</span>
          </span>
          <button
            onClick={() => { logout(); navigate("/"); }}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-xl border border-border text-foreground hover:bg-muted hover:text-destructive transition-all duration-200"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Log Out</span>
          </button>
        </div>
      </nav>

      <main className="flex-1">
        {isLoading && (
          <section className="py-24 px-4 text-center">
            <div className="max-w-md mx-auto">
              <div className="inline-block w-14 h-14 border-4 border-primary border-t-transparent rounded-full animate-spin mb-6" />
              <p className="text-xl font-display font-semibold text-foreground">
                Analysing your profile…
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                Our AI is finding your best career matches
              </p>
            </div>
          </section>
        )}

        {apiError && !isLoading && (
          <section className="py-6 px-4">
            <div className="max-w-xl mx-auto rounded-xl border border-destructive/50 bg-destructive/10 p-4 text-destructive text-sm flex items-start justify-between gap-4">
              <span>
                <strong>Error: </strong>
                {apiError}
              </span>
              <button
                className="shrink-0 underline text-xs hover:text-destructive/70 transition-colors"
                onClick={() => setApiError(null)}
              >
                Dismiss
              </button>
            </div>
          </section>
        )}

        {!isLoading && !results && (
          <AssessmentForm onSubmit={handleSubmit} />
        )}

        {!isLoading && results && (
          <CareerResults
            careers={results.careers}
            universities={results.universities}
            jobs={results.jobs}
            onReset={handleReset}
          />
        )}
      </main>

      <footer className="py-6 text-center text-sm text-muted-foreground border-t border-border">
        <p>Career Path Recommender — Helping students find their future.</p>
      </footer>
    </div>
  );
};

export default Assessment;
