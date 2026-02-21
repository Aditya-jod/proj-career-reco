import { useState } from "react";
import {
  TrendingUp,
  DollarSign,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  GraduationCap,
  Briefcase,
  ExternalLink,
  MapPin,
} from "lucide-react";
import type { CareerPath, University, Job } from "@/data/careerData";

interface CareerResultsProps {
  careers: CareerPath[];
  universities: University[];
  jobs: Job[];
  onReset: () => void;
}

const CareerResults = ({ careers, universities, jobs, onReset }: CareerResultsProps) => {
  const [expanded, setExpanded] = useState<string | null>(careers[0]?.title || null);

  return (
    <section className="py-20 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <span className="chip-accent mb-3 inline-block">✨ Your Results</span>
          <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-3">
            Recommended Career Paths
          </h2>
          <p className="text-muted-foreground">
            Based on your profile, here are the best matches for you.
          </p>
        </div>

        {/* ── Career Cards ── */}
        <div className="space-y-4 mb-16">
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
                    {career.skills.length > 0 && (
                      <div className="mb-6">
                        <p className="text-sm font-medium text-foreground mb-2">Key Skills</p>
                        <div className="flex flex-wrap gap-2">
                          {career.skills.map((skill) => (
                            <span key={skill} className="chip">{skill}</span>
                          ))}
                        </div>
                      </div>
                    )}

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

        {/* ── University Recommendations ── */}
        {universities.length > 0 && (
          <div className="mb-16">
            <div className="flex items-center gap-3 mb-6">
              <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-accent/10">
                <GraduationCap className="w-5 h-5 text-accent" />
              </div>
              <div>
                <h3 className="text-xl font-display font-bold text-foreground">
                  Recommended Universities
                </h3>
                <p className="text-sm text-muted-foreground">
                  Top {universities.length} universities matching your profile
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {universities.map((uni, idx) => (
                <div
                  key={`${uni.name}-${idx}`}
                  className="glass-card p-5 flex flex-col gap-3 animate-fade-up"
                  style={{ animationDelay: `${idx * 0.07}s` }}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <h4 className="font-semibold text-foreground text-sm leading-snug line-clamp-2">
                        {uni.name}
                      </h4>
                      <div className="flex items-center gap-1 mt-1 text-xs text-muted-foreground">
                        <MapPin className="w-3 h-3 shrink-0" />
                        <span className="truncate">
                          {[uni.district, uni.state, uni.country]
                            .filter(Boolean)
                            .join(", ")}
                        </span>
                      </div>
                    </div>
                    {uni.website && uni.website !== "N/A" && (
                      <a
                        href={uni.website.startsWith("http") ? uni.website : `https://${uni.website}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="shrink-0 text-primary hover:text-primary/70 transition-colors"
                        title="Visit website"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    )}
                  </div>

                  {/* Score bar */}
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-muted-foreground">Match score</span>
                      <span className="font-medium text-accent">
                        {(uni.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-border overflow-hidden">
                      <div
                        className="h-full rounded-full bg-accent transition-all duration-700"
                        style={{ width: `${uni.score * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Job Recommendations ── */}
        {jobs.length > 0 && (
          <div className="mb-16">
            <div className="flex items-center gap-3 mb-6">
              <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-primary/10">
                <Briefcase className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-display font-bold text-foreground">
                  Recommended Roles
                </h3>
                <p className="text-sm text-muted-foreground">
                  Top {jobs.length} job roles that fit your strengths
                </p>
              </div>
            </div>

            <div className="glass-card p-6">
              <div className="space-y-3">
                {jobs.map((job, idx) => (
                  <div
                    key={`${job.title}-${idx}`}
                    className="flex items-center justify-between gap-4 py-2 border-b border-border/40 last:border-0 animate-fade-up"
                    style={{ animationDelay: `${idx * 0.05}s` }}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-bold text-muted-foreground w-5 shrink-0">
                        {idx + 1}.
                      </span>
                      <span className="text-sm font-medium text-foreground">{job.title}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <div className="w-20 h-1.5 rounded-full bg-border overflow-hidden hidden sm:block">
                        <div
                          className="h-full rounded-full bg-primary transition-all duration-700"
                          style={{ width: `${job.score * 100}%` }}
                        />
                      </div>
                      <span className="text-xs font-medium text-primary w-10 text-right">
                        {(job.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Reset */}
        <div className="text-center mt-4">
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
