import { useState } from "react";
import { TrendingUp, DollarSign, ChevronDown, ChevronUp, RotateCcw } from "lucide-react";
import type { CareerPath } from "@/data/careerData";

interface CareerResultsProps {
  careers: CareerPath[];
  onReset: () => void;
}

const CareerResults = ({ careers, onReset }: CareerResultsProps) => {
  const [expanded, setExpanded] = useState<string | null>(careers[0]?.title || null);

  return (
    <section className="py-20 px-4">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <span className="chip-accent mb-3 inline-block">✨ Your Results</span>
          <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-3">
            Recommended Career Paths
          </h2>
          <p className="text-muted-foreground">Based on your profile, here are the best matches for you.</p>
        </div>

        <div className="space-y-4">
          {careers.map((career, idx) => {
            const isOpen = expanded === career.title;
            return (
              <div
                key={career.title}
                className="glass-card overflow-hidden animate-fade-up"
                style={{ animationDelay: `${idx * 0.1}s` }}
              >
                {/* Header */}
                <button
                  onClick={() => setExpanded(isOpen ? null : career.title)}
                  className="w-full flex items-center justify-between p-6 text-left"
                >
                  <div className="flex items-center gap-4">
                    <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/10 text-primary font-display font-bold text-lg">
                      {idx + 1}
                    </div>
                    <div>
                      <h3 className="text-lg font-display font-semibold text-foreground">
                        {career.title}
                      </h3>
                      <p className="text-sm text-muted-foreground">{career.field}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="hidden sm:flex items-center gap-1 text-sm font-medium text-primary">
                      {career.matchScore}% Match
                    </div>
                    <div className="w-16 h-2 rounded-full bg-border overflow-hidden hidden sm:block">
                      <div
                        className="h-full rounded-full bg-primary transition-all duration-700"
                        style={{ width: `${career.matchScore}%` }}
                      />
                    </div>
                    {isOpen ? (
                      <ChevronUp className="w-5 h-5 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-muted-foreground" />
                    )}
                  </div>
                </button>

                {/* Expanded Content */}
                {isOpen && (
                  <div className="px-6 pb-6 border-t border-border/50">
                    <p className="text-muted-foreground mt-4 mb-6">{career.description}</p>

                    {/* Stats */}
                    <div className="flex flex-wrap gap-4 mb-6">
                      <div className="flex items-center gap-2 text-sm">
                        <DollarSign className="w-4 h-4 text-accent" />
                        <span className="text-foreground font-medium">{career.avgSalary}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <TrendingUp className="w-4 h-4 text-primary" />
                        <span className="text-foreground font-medium">Growth: {career.growth}</span>
                      </div>
                    </div>

                    {/* Skills */}
                    <div className="mb-6">
                      <p className="text-sm font-medium text-foreground mb-2">Key Skills</p>
                      <div className="flex flex-wrap gap-2">
                        {career.skills.map((skill) => (
                          <span key={skill} className="chip">{skill}</span>
                        ))}
                      </div>
                    </div>

                    {/* Pathway */}
                    <div>
                      <p className="text-sm font-medium text-foreground mb-3">Career Pathway</p>
                      <div className="space-y-0">
                        {career.pathway.map((step, i) => (
                          <div key={i} className="flex items-start gap-3">
                            <div className="flex flex-col items-center">
                              <div className="w-3 h-3 rounded-full bg-primary mt-1.5" />
                              {i < career.pathway.length - 1 && (
                                <div className="step-connector" />
                              )}
                            </div>
                            <div className="pb-4">
                              <p className="text-sm font-semibold text-foreground">{step.stage}</p>
                              <p className="text-sm text-muted-foreground">{step.detail}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="text-center mt-10">
          <button
            onClick={onReset}
            className="inline-flex items-center gap-2 px-6 py-3 border border-border rounded-xl
              text-muted-foreground hover:text-foreground hover:border-primary/40 transition-all duration-200"
          >
            <RotateCcw className="w-4 h-4" /> Retake Assessment
          </button>
        </div>
      </div>
    </section>
  );
};

export default CareerResults;
