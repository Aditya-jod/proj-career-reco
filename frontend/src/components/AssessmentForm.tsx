import { useState, useRef, KeyboardEvent } from "react";
import { ChevronRight, ChevronLeft, Sparkles, MapPin, X } from "lucide-react";
import { SUGGESTIONS, type SuggestionsData } from "@/data/careerData";

// Suggestion lists are hardcoded in careerData.ts — no DB dependency.

interface ScoreFields {
  mathematics: number;
  science: number;
  language_arts: number;
  social_studies: number;
  logical_reasoning: number;
  creativity: number;
  communication: number;
  leadership: number;
  social_skills: number;
}

interface AssessmentFormProps {
  onSubmit: (data: {
    academics: string;
    interests: string[];
    skills: string[];
    hobbies: string[];
    scores: ScoreFields;
    preferredLocation: string;
  }) => void;
}

// academicStreams come from hardcoded SUGGESTIONS constant
// (no API call needed)

const defaultScores: ScoreFields = {
  mathematics: 70,
  science: 70,
  language_arts: 70,
  social_studies: 70,
  logical_reasoning: 70,
  creativity: 70,
  communication: 70,
  leadership: 70,
  social_skills: 70,
};

const scoreLabels: { key: keyof ScoreFields; label: string; emoji: string }[] = [
  { key: "mathematics", label: "Mathematics", emoji: "🔢" },
  { key: "science", label: "Science", emoji: "🔬" },
  { key: "language_arts", label: "Language Arts", emoji: "📖" },
  { key: "social_studies", label: "Social Studies", emoji: "🌍" },
  { key: "logical_reasoning", label: "Logical Reasoning", emoji: "🧩" },
  { key: "creativity", label: "Creativity", emoji: "🎨" },
  { key: "communication", label: "Communication", emoji: "💬" },
  { key: "leadership", label: "Leadership", emoji: "👥" },
  { key: "social_skills", label: "Social Skills", emoji: "🤝" },
];

// ── TagInput ─────────────────────────────────────────────────────────────────
interface TagInputProps {
  tags: string[];
  suggestions: string[];
  placeholder: string;
  color: "primary" | "accent";
  onAdd: (tag: string) => void;
  onRemove: (tag: string) => void;
}

const TagInput = ({ tags, suggestions, placeholder, color, onAdd, onRemove }: TagInputProps) => {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const commit = (raw: string) => {
    raw.split(",").map((t) => t.trim()).filter(Boolean).forEach((t) => {
      if (!tags.some((x) => x.toLowerCase() === t.toLowerCase())) onAdd(t);
    });
    setInput("");
  };

  const handleKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") { e.preventDefault(); commit(input); }
    if (e.key === "Backspace" && input === "" && tags.length > 0) onRemove(tags[tags.length - 1]);
  };

  const chipClass = color === "primary"
    ? "bg-primary text-primary-foreground"
    : "bg-accent text-accent-foreground";

  const filtered = suggestions.filter(
    (s) => !tags.some((t) => t.toLowerCase() === s.toLowerCase())
  );

  return (
    <div className="space-y-3">
      {/* Tag chips + text input */}
      <div
        className="flex flex-wrap gap-2 p-3 rounded-xl border border-border bg-background cursor-text min-h-[52px]"
        onClick={() => inputRef.current?.focus()}
      >
        {tags.map((tag) => (
          <span key={tag} className={`flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${chipClass}`}>
            {tag}
            <button type="button" onClick={(e) => { e.stopPropagation(); onRemove(tag); }} className="opacity-70 hover:opacity-100">
              <X className="w-3 h-3" />
            </button>
          </span>
        ))}
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          onBlur={() => { if (input.trim()) commit(input); }}
          placeholder={tags.length === 0 ? placeholder : ""}
          className="flex-1 min-w-[140px] bg-transparent outline-none text-sm text-foreground placeholder:text-muted-foreground"
        />
      </div>
      <p className="text-xs text-muted-foreground">Type anything and press Enter (or comma to separate multiple). Or click a suggestion:</p>
      {/* Quick-pick suggestions */}
      <div className="flex flex-wrap gap-2">
        {filtered.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onAdd(s)}
            className="px-3 py-1 rounded-full border border-border text-sm text-muted-foreground hover:border-primary/50 hover:text-foreground hover:bg-muted transition-all"
          >
            + {s}
          </button>
        ))}
      </div>
    </div>
  );
};

// ── AssessmentForm ────────────────────────────────────────────────────────────
const AssessmentForm = ({ onSubmit }: AssessmentFormProps) => {
  const [step, setStep] = useState(0);
  const [academics, setAcademics] = useState("");
  const [interests, setInterests] = useState<string[]>([]);
  const [skills, setSkills] = useState<string[]>([]);
  const [hobbies, setHobbies] = useState<string[]>([]);
  const [scores, setScores] = useState<ScoreFields>(defaultScores);
  const [preferredLocation, setPreferredLocation] = useState("");

  // ── Hardcoded suggestions (no backend fetch needed) ────────────────
  const suggestions: SuggestionsData = SUGGESTIONS;

  const setScore = (key: keyof ScoreFields, value: number) => {
    setScores((prev) => ({ ...prev, [key]: value }));
  };

  const addTag = (list: string[], setter: React.Dispatch<React.SetStateAction<string[]>>, tag: string) => {
    if (!list.some((t) => t.toLowerCase() === tag.toLowerCase())) setter([...list, tag]);
  };
  const removeTag = (list: string[], setter: React.Dispatch<React.SetStateAction<string[]>>, tag: string) => {
    setter(list.filter((t) => t !== tag));
  };

  const steps = [
    {
      title: "Academic Stream",
      subtitle: "What's your academic focus area?",
      content: (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {suggestions.academic_streams.map((stream) => (
            <button
              key={stream}
              onClick={() => setAcademics(stream)}
              className={`p-4 rounded-xl border-2 text-left font-medium transition-all duration-200
                ${academics === stream
                  ? "border-primary bg-primary/10 text-primary shadow-md"
                  : "border-border hover:border-primary/40 text-foreground hover:bg-muted"
                }`}
            >
              {stream}
            </button>
          ))}
        </div>
      ),
      isValid: () => academics !== "",
    },
    {
      title: "Your Interests",
      subtitle: "Type your interests or pick from suggestions below",
      content: (
        <TagInput
          tags={interests}
          suggestions={suggestions.interests}
          placeholder="e.g. Technology, Healthcare, Music…"
          color="primary"
          onAdd={(t) => addTag(interests, setInterests, t)}
          onRemove={(t) => removeTag(interests, setInterests, t)}
        />
      ),
      isValid: () => interests.length >= 1,
    },
    {
      title: "Your Skills",
      subtitle: "Type your skills or pick from suggestions below",
      content: (
        <TagInput
          tags={skills}
          suggestions={suggestions.skills}
          placeholder="e.g. Problem Solving, Programming…"
          color="primary"
          onAdd={(t) => addTag(skills, setSkills, t)}
          onRemove={(t) => removeTag(skills, setSkills, t)}
        />
      ),
      isValid: () => skills.length >= 1,
    },
    {
      title: "Your Hobbies",
      subtitle: "Type your hobbies or pick from suggestions below",
      content: (
        <TagInput
          tags={hobbies}
          suggestions={suggestions.hobbies}
          placeholder="e.g. Reading, Chess, Robotics…"
          color="accent"
          onAdd={(t) => addTag(hobbies, setHobbies, t)}
          onRemove={(t) => removeTag(hobbies, setHobbies, t)}
        />
      ),
      isValid: () => hobbies.length >= 1,
    },
    {
      title: "Scores & Location",
      subtitle: "Rate your strengths (0–100) and enter your preferred study location",
      content: (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-4">
            {scoreLabels.map(({ key, label, emoji }) => (
              <div key={key}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium text-foreground">
                    {emoji} {label}
                  </span>
                  <span className="text-sm font-bold text-primary w-8 text-right">
                    {scores[key]}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={scores[key]}
                  onChange={(e) => setScore(key, Number(e.target.value))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-primary bg-border"
                />
              </div>
            ))}
          </div>

          <div className="mt-4">
            <label className="flex items-center gap-2 text-sm font-medium text-foreground mb-2">
              <MapPin className="w-4 h-4 text-primary" />
              Preferred Study Location{" "}
              <span className="text-muted-foreground font-normal">(optional)</span>
            </label>
            <input
              type="text"
              placeholder="e.g. India, United States, Germany…"
              value={preferredLocation}
              onChange={(e) => setPreferredLocation(e.target.value)}
              className="w-full px-4 py-2.5 rounded-xl border border-border bg-background text-foreground
                focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary
                placeholder:text-muted-foreground transition-all"
            />
          </div>
        </div>
      ),
      isValid: () => true, // always valid — scores have defaults, location is optional
    },
  ];

  const currentStep = steps[step];
  const isLast = step === steps.length - 1;

  return (
    <section id="assessment" className="py-20 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Progress */}
        <div className="flex items-center gap-2 mb-8">
          {steps.map((_, i) => (
            <div
              key={i}
              className={`h-1.5 flex-1 rounded-full transition-all duration-300 ${
                i <= step ? "bg-primary" : "bg-border"
              }`}
            />
          ))}
        </div>

        {/* Step card */}
        <div className="glass-card p-8">
          <p className="text-sm font-medium text-muted-foreground mb-1">
            Step {step + 1} of {steps.length}
          </p>
          <h2 className="text-2xl font-display font-bold text-foreground mb-1">
            {currentStep.title}
          </h2>
          <p className="text-muted-foreground mb-6">{currentStep.subtitle}</p>

          {currentStep.content}

          {/* Navigation */}
          <div className="flex justify-between mt-8">
            <button
              onClick={() => setStep(Math.max(0, step - 1))}
              disabled={step === 0}
              className="flex items-center gap-1 px-4 py-2 text-muted-foreground hover:text-foreground disabled:opacity-30 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" /> Back
            </button>

            {isLast ? (
              <button
                disabled={!currentStep.isValid()}
                onClick={() =>
                  onSubmit({ academics, interests, skills, hobbies, scores, preferredLocation })
                }
                className="flex items-center gap-2 px-6 py-3 bg-accent text-accent-foreground font-semibold rounded-xl
                  hover:shadow-lg disabled:opacity-40 transition-all duration-200 hover:scale-105"
              >
                <Sparkles className="w-4 h-4" /> Get Recommendations
              </button>
            ) : (
              <button
                disabled={!currentStep.isValid()}
                onClick={() => setStep(step + 1)}
                className="flex items-center gap-1 px-6 py-3 bg-primary text-primary-foreground font-semibold rounded-xl
                  hover:shadow-lg disabled:opacity-40 transition-all duration-200"
              >
                Next <ChevronRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default AssessmentForm;
