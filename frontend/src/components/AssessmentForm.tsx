import { useState } from "react";
import { ChevronRight, ChevronLeft, Sparkles } from "lucide-react";
import { interestOptions, skillOptions, hobbyOptions } from "@/data/careerData";

interface AssessmentFormProps {
  onSubmit: (data: {
    academics: string;
    interests: string[];
    skills: string[];
    hobbies: string[];
  }) => void;
}

const academicStreams = [
  "Science (PCM)",
  "Science (PCB)",
  "Commerce",
  "Arts / Humanities",
];

const AssessmentForm = ({ onSubmit }: AssessmentFormProps) => {
  const [step, setStep] = useState(0);
  const [academics, setAcademics] = useState("");
  const [interests, setInterests] = useState<string[]>([]);
  const [skills, setSkills] = useState<string[]>([]);
  const [hobbies, setHobbies] = useState<string[]>([]);

  const toggleItem = (
    list: string[],
    setter: React.Dispatch<React.SetStateAction<string[]>>,
    item: string
  ) => {
    setter(list.includes(item) ? list.filter((i) => i !== item) : [...list, item]);
  };

  const steps = [
    {
      title: "Academic Stream",
      subtitle: "What's your Class 12 stream?",
      content: (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {academicStreams.map((stream) => (
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
      subtitle: "Select areas that excite you (pick 2-4)",
      content: (
        <div className="flex flex-wrap gap-2">
          {interestOptions.map((item) => (
            <button
              key={item}
              onClick={() => toggleItem(interests, setInterests, item)}
              className={`px-4 py-2 rounded-full border text-sm font-medium transition-all duration-200
                ${interests.includes(item)
                  ? "border-primary bg-primary text-primary-foreground shadow-md"
                  : "border-border hover:border-primary/40 text-foreground hover:bg-muted"
                }`}
            >
              {item}
            </button>
          ))}
        </div>
      ),
      isValid: () => interests.length >= 1,
    },
    {
      title: "Your Skills",
      subtitle: "What are you naturally good at? (pick 2-4)",
      content: (
        <div className="flex flex-wrap gap-2">
          {skillOptions.map((item) => (
            <button
              key={item}
              onClick={() => toggleItem(skills, setSkills, item)}
              className={`px-4 py-2 rounded-full border text-sm font-medium transition-all duration-200
                ${skills.includes(item)
                  ? "border-primary bg-primary text-primary-foreground shadow-md"
                  : "border-border hover:border-primary/40 text-foreground hover:bg-muted"
                }`}
            >
              {item}
            </button>
          ))}
        </div>
      ),
      isValid: () => skills.length >= 1,
    },
    {
      title: "Your Hobbies",
      subtitle: "What do you love doing in your free time? (pick 2-4)",
      content: (
        <div className="flex flex-wrap gap-2">
          {hobbyOptions.map((item) => (
            <button
              key={item}
              onClick={() => toggleItem(hobbies, setHobbies, item)}
              className={`px-4 py-2 rounded-full border text-sm font-medium transition-all duration-200
                ${hobbies.includes(item)
                  ? "border-accent bg-accent text-accent-foreground shadow-md"
                  : "border-border hover:border-accent/40 text-foreground hover:bg-muted"
                }`}
            >
              {item}
            </button>
          ))}
        </div>
      ),
      isValid: () => hobbies.length >= 1,
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
                  onSubmit({ academics, interests, skills, hobbies })
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
