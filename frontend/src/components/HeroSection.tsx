import { ArrowRight, Compass, Brain, Target, Rocket, Users } from "lucide-react";
import { Link } from "react-router-dom";
import heroBg from "@/assets/CPR project image.png";

interface HeroSectionProps {
  onStart: () => void;
}

const features = [
  { icon: Brain, title: "AI Analysis", desc: "Smart matching based on your unique profile" },
  { icon: Target, title: "Precision Match", desc: "Tailored paths from Class 12 to career" },
  { icon: Rocket, title: "Growth Roadmap", desc: "Step-by-step pathway to professional success" },
];

const steps = [
  {
    step: "01",
    title: "Share Your Profile",
    desc: "Tell us about your stream, interests, skills, and hobbies — it only takes 2 minutes.",
    icon: Users,
  },
  {
    step: "02",
    title: "AI Analyzes",
    desc: "Our algorithm evaluates your strengths against 50+ career paths and industry data.",
    icon: Brain,
  },
  {
    step: "03",
    title: "Get Your Roadmap",
    desc: "Receive ranked career matches with detailed pathways from Class 12 to professional success.",
    icon: Rocket,
  },
];

const HeroSection = ({ onStart }: HeroSectionProps) => {
  return (
    <>
      {/* Hero */}
      <section className="relative min-h-screen flex items-center overflow-hidden">
        {/* Nav bar with auth buttons */}
        <nav className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-6 py-4 max-w-7xl mx-auto">
          <span className="font-display font-bold text-xl text-foreground">CareerPath</span>
          <div className="flex items-center gap-3">
            <Link
              to="/login"
              className="px-4 py-2 text-sm font-medium rounded-xl border border-primary text-foreground hover:bg-primary/10 hover:text-primary transition-all duration-300 hover:shadow-[0_0_20px_hsl(var(--primary)/0.3)]"
            >
              Log In
            </Link>
            <Link
              to="/signup"
              className="px-4 py-2 text-sm font-medium rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-300 hover:shadow-[0_0_25px_hsl(var(--primary)/0.5)] hover:scale-105"
            >
              Sign Up
            </Link>
          </div>
        </nav>

        {/* Background image with overlays */}
        <div className="absolute inset-0">
          <img src={heroBg} alt="" className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-br from-background/40 via-background/50 to-primary/10" />
          <div className="absolute inset-0 bg-gradient-to-t from-background/20 via-background/30 to-transparent" />
        </div>

        {/* Animated orbs */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute w-72 h-72 rounded-full bg-primary/10 blur-[100px] animate-pulse-slow top-20 -left-20" />
          <div className="absolute w-96 h-96 rounded-full bg-accent/10 blur-[120px] animate-pulse-slow bottom-20 right-10" style={{ animationDelay: "2s" }} />
          <div className="absolute w-48 h-48 rounded-full bg-glow/15 blur-[80px] animate-float top-1/2 left-1/3" />
        </div>

        {/* Content */}
        <div className="relative z-10 w-full max-w-7xl mx-auto px-6 py-24 grid lg:grid-cols-2 gap-16 items-center">
          {/* Left column */}
          <div>
            <div className="animate-fade-up">
              <span className="inline-flex items-center gap-2 px-4 py-1.5 mb-8 rounded-full border border-glow/30 bg-glow/10 text-sm font-medium text-primary backdrop-blur-sm">
                <Compass className="w-4 h-4" />
                AI-Powered Career Guidance
              </span>
            </div>

            <h1 className="animate-fade-up-delay-1 text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-display font-bold leading-[1.1] tracking-tight mb-6">
              <span className="text-foreground">Find Your</span>
              <br />
              <span className="bg-gradient-to-r from-primary via-glow to-accent bg-clip-text text-transparent">Dream Career</span>
              <br />
              <span className="text-foreground">Path Today</span>
            </h1>

            <p className="animate-fade-up-delay-2 text-lg text-muted-foreground max-w-lg mb-10 leading-relaxed">
              Answer a few questions about your interests, skills, and academics.
              Get personalized career recommendations — from Class 12 to professional success.
            </p>

            <div className="animate-fade-up-delay-3 flex flex-wrap items-center gap-4">
              <button
                onClick={onStart}
                className="group relative px-8 py-4 bg-gradient-to-r from-primary to-glow text-primary-foreground font-display font-semibold text-lg rounded-2xl overflow-hidden transition-all duration-300 hover:scale-110 hover:shadow-[0_0_50px_hsl(var(--glow)/0.6)] before:absolute before:inset-0 before:rounded-2xl before:bg-gradient-to-r before:from-primary/50 before:to-glow/50 before:opacity-0 before:transition-opacity before:duration-300 hover:before:opacity-100"
              >
                <span className="relative z-10 flex items-center gap-2">
                  Start Free Assessment
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-2 transition-transform duration-300" />
                </span>
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-primary via-glow to-accent opacity-0 group-hover:opacity-20 transition-opacity duration-300 blur" />
              </button>
            </div>
          </div>

          {/* Right column - Feature cards */}
          <div className="hidden lg:flex flex-col gap-4">
            {features.map((feat, i) => (
              <div
                key={feat.title}
                className="animate-fade-up group relative glass-card p-6 flex items-start gap-4 hover:border-primary/60 transition-all duration-300 hover:scale-[1.05] hover:shadow-[0_0_30px_hsl(var(--primary)/0.3)] cursor-default overflow-hidden"
                style={{ animationDelay: `${0.3 + i * 0.15}s` }}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-glow/20 flex items-center justify-center group-hover:scale-125 transition-transform duration-300 group-hover:shadow-[0_0_25px_hsl(var(--primary)/0.4)]">
                  <feat.icon className="w-6 h-6 text-primary" />
                </div>
                <div className="relative z-10">
                  <h3 className="font-display font-semibold text-foreground mb-1 group-hover:text-primary transition-colors duration-300">{feat.title}</h3>
                  <p className="text-sm text-muted-foreground group-hover:text-foreground/80 transition-colors duration-300">{feat.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom fade */}
        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
      </section>

      {/* How it works section */}
      <section className="py-20 px-6 bg-background">
        <div className="max-w-5xl mx-auto text-center mb-14">
          <span className="chip mb-4 inline-block">How It Works</span>
          <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4">Three Simple Steps to Your Future</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Our AI analyzes your unique profile and matches you with career paths where you're most likely to thrive.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-3">
          {steps.map((item, i) => (
            <div
              key={item.step}
              className="group relative glass-card p-8 text-center hover:border-primary/60 transition-all duration-300 hover:shadow-[0_0_35px_hsl(var(--primary)/0.3)] hover:-translate-y-2 overflow-hidden"
              style={{ animationDelay: `${i * 0.2}s` }}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-gradient-to-r from-primary to-glow text-primary-foreground flex items-center justify-center text-xs font-bold font-display group-hover:scale-125 transition-transform duration-300 group-hover:shadow-[0_0_20px_hsl(var(--primary)/0.5)]">
                {item.step}
              </div>
              <div className="relative z-10 w-14 h-14 mx-auto mb-5 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center group-hover:scale-125 transition-transform duration-300 group-hover:shadow-[0_0_25px_hsl(var(--primary)/0.4)]">
                <item.icon className="w-7 h-7 text-primary" />
              </div>
              <h3 className="relative z-10 font-display font-semibold text-lg text-foreground mb-2 group-hover:text-primary transition-colors duration-300">{item.title}</h3>
              <p className="relative z-10 text-sm text-muted-foreground leading-relaxed group-hover:text-foreground/80 transition-colors duration-300">{item.desc}</p>
            </div>
          ))}
        </div>

        {/* CTA */}
        <div className="text-center mt-14">
          <button
            onClick={onStart}
            className="group px-8 py-4 bg-accent text-accent-foreground font-display font-semibold text-lg rounded-2xl hover:shadow-[0_0_30px_hsl(var(--accent)/0.4)] transition-all duration-300 hover:scale-105"
          >
            Begin Your Journey
            <ArrowRight className="inline-block ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </section>
    </>
  );
};

export default HeroSection;
