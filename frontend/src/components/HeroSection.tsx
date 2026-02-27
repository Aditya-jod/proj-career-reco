import { useRef, useState, useEffect, useCallback, type ReactNode } from "react";
import {
  ArrowRight,
  Compass,
  Brain,
  Rocket,
  Users,
  LogOut,
  ChevronDown,
  Cpu,
  Heart,
  TrendingUp,
  Palette,
  GraduationCap,
  Scale,
  Wrench,
  Star,
  Sparkles,
  BarChart3,
  Globe,
  Target,
} from "lucide-react";
import { Link } from "react-router-dom";
import heroBg from "@/assets/CPR project image.png";
import { useAuth } from "@/context/AuthContext";

/* ─── Parallax scroll hook ───────────────────────────────────────────── */

function useParallax(speed = 0.3, maxShift = 80) {
  const ref = useRef<HTMLDivElement>(null);
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    let ticking = false;
    const handleScroll = () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(() => {
          if (ref.current) {
            const rect = ref.current.getBoundingClientRect();
            const viewH = window.innerHeight;
            if (rect.bottom > -200 && rect.top < viewH + 200) {
              // Centre-based: 0 when element centre aligns with viewport centre
              const centre = rect.top + rect.height / 2;
              const raw = (centre - viewH / 2) * speed;
              // Clamp to prevent over-shifting
              setOffset(Math.max(-maxShift, Math.min(maxShift, raw)));
            }
          }
          ticking = false;
        });
      }
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, [speed, maxShift]);

  return { ref, offset };
}

/* ─── Types ──────────────────────────────────────────────────────────── */

interface HeroSectionProps {
  onStart: () => void;
}

/* ─── Scroll-triggered reveal wrapper ────────────────────────────────── */

function ScrollReveal({
  children,
  className = "",
  delay = 0,
  threshold = 0.15,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
  threshold?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.unobserve(el);
        }
      },
      { threshold },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [threshold]);

  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(32px)",
        transition: `opacity 0.7s cubic-bezier(.22,1,.36,1) ${delay}s, transform 0.7s cubic-bezier(.22,1,.36,1) ${delay}s`,
      }}
    >
      {children}
    </div>
  );
}

/* ─── Animated counter ───────────────────────────────────────────────── */

function CountUp({
  target,
  suffix = "",
  duration = 2000,
  decimals = 0,
}: {
  target: number;
  suffix?: string;
  duration?: number;
  decimals?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [count, setCount] = useState(0);
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started) {
          setStarted(true);
          observer.unobserve(el);
        }
      },
      { threshold: 0.5 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [started]);

  useEffect(() => {
    if (!started) return;
    const start = performance.now();
    let raf: number;
    const multiplier = Math.pow(10, decimals);
    const intTarget = Math.round(target * multiplier);

    const step = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      setCount(Math.floor(eased * intTarget));
      if (progress < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [started, target, duration, decimals]);

  const display =
    decimals > 0
      ? (count / Math.pow(10, decimals)).toFixed(decimals)
      : count.toLocaleString();

  return (
    <span ref={ref}>
      {display}
      {suffix}
    </span>
  );
}

/* ─── Data ───────────────────────────────────────────────────────────── */

const stats = [
  { value: 8, suffix: "", label: "Career Fields", icon: Compass, decimals: 0 },
  { value: 48, suffix: "K+", label: "Universities", icon: Globe, decimals: 0 },
  { value: 1.6, suffix: "M+", label: "Jobs Analyzed", icon: BarChart3, decimals: 1 },
  { value: 95, suffix: "%", label: "Match Accuracy", icon: Target, decimals: 0 },
];

const careerFields = [
  { id: "STEM", title: "STEM", icon: Cpu, desc: "Science, Technology, Engineering & Mathematics" },
  { id: "Healthcare", title: "Healthcare", icon: Heart, desc: "Medicine & Life Sciences" },
  { id: "Business_Finance", title: "Business & Finance", icon: TrendingUp, desc: "Commerce, Economics & Management" },
  { id: "Arts_Media", title: "Arts & Media", icon: Palette, desc: "Design, Creative Arts & Communication" },
  { id: "Education", title: "Education", icon: GraduationCap, desc: "Teaching, Training & Research" },
  { id: "Social_Services", title: "Social Services", icon: Users, desc: "Counseling, Welfare & Community" },
  { id: "Government_Law", title: "Government & Law", icon: Scale, desc: "Legal, Policy & Public Administration" },
  { id: "Trades_Manufacturing", title: "Trades", icon: Wrench, desc: "Skilled Trades & Manufacturing" },
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
    desc: "Our algorithm evaluates your strengths against career paths and real industry data.",
    icon: Brain,
  },
  {
    step: "03",
    title: "Get Your Roadmap",
    desc: "Receive ranked career matches with university recommendations and detailed pathways.",
    icon: Rocket,
  },
];

const testimonials = [
  {
    name: "Priya Sharma",
    role: "B.Tech Computer Science",
    quote:
      "CareerPath helped me realize data science was the perfect fit. The AI analysis was incredibly accurate!",
    initials: "PS",
    rating: 5,
  },
  {
    name: "Rahul Patel",
    role: "Commerce Graduate",
    quote:
      "I was confused between finance and marketing. The detailed roadmap gave me clarity and confidence.",
    initials: "RP",
    rating: 5,
  },
  {
    name: "Ananya Reddy",
    role: "Science Student (PCB)",
    quote:
      "The university recommendations were spot-on. I discovered amazing colleges I hadn't even considered!",
    initials: "AR",
    rating: 5,
  },
];

/* ─── Component ──────────────────────────────────────────────────────── */

const HeroSection = ({ onStart }: HeroSectionProps) => {
  const { isAuthenticated, user, logout } = useAuth();

  const scrollToHowItWorks = useCallback(() => {
    document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" });
  }, []);

  // Parallax for hero background
  const heroParallax = useParallax(0.25);
  // Parallax for How It Works orbs
  const howItWorksParallax = useParallax(0.15);
  // Parallax for closing CTA orbs
  const ctaParallax = useParallax(0.2);

  return (
    <>
      {/* ════════════════════════ HERO ════════════════════════ */}
      <section className="relative min-h-screen flex items-center overflow-hidden">
        {/* Nav */}
        <nav className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-6 py-4 max-w-7xl mx-auto">
          <span className="font-display font-bold text-xl text-foreground flex items-center gap-2">
            <Compass className="w-6 h-6 text-primary" />
            CareerPath
          </span>
          <div className="flex items-center gap-3">
            {isAuthenticated ? (
              <>
                <span className="text-sm text-muted-foreground hidden sm:inline">
                  Hi,{" "}
                  <span className="font-medium text-foreground">{user?.name}</span>
                </span>
                <button
                  onClick={logout}
                  className="flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-xl border border-border text-foreground hover:bg-muted hover:text-destructive transition-all duration-300"
                >
                  <LogOut className="w-3.5 h-3.5" />
                  Log Out
                </button>
              </>
            ) : (
              <>
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
              </>
            )}
          </div>
        </nav>

        {/* Background image + overlays */}
        <div className="absolute inset-0">
          {/* Parallax image — oversized container so translate never reveals gaps */}
          <div
            ref={heroParallax.ref}
            className="absolute inset-x-0 -top-[12%] -bottom-[12%] will-change-transform"
            style={{ transform: `translateY(${heroParallax.offset}px)` }}
          >
            <img src={heroBg} alt="" className="w-full h-full object-cover" />
          </div>
          {/* Fixed overlays — stay aligned with section, not parallax */}
          <div className="absolute inset-0 bg-gradient-to-br from-background/40 via-background/25 to-primary/5" />
          <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-background/20 to-transparent" />
        </div>

        {/* Animated orbs */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute w-72 h-72 rounded-full bg-primary/10 blur-[100px] animate-pulse-slow top-20 -left-20" />
          <div
            className="absolute w-96 h-96 rounded-full bg-accent/10 blur-[120px] animate-pulse-slow bottom-20 right-10"
            style={{ animationDelay: "2s" }}
          />
          <div className="absolute w-48 h-48 rounded-full bg-glow/15 blur-[80px] animate-float top-1/2 left-1/3" />
        </div>

        {/* Hero content — centred */}
        <div className="relative z-10 w-full max-w-7xl mx-auto px-6 pt-28 pb-20 flex flex-col items-center text-center">
          <div className="animate-fade-up">
            <span className="inline-flex items-center gap-2 px-4 py-1.5 mb-6 rounded-full border border-glow/30 bg-glow/10 text-sm font-medium text-primary backdrop-blur-sm">
              <Sparkles className="w-4 h-4" />
              AI-Powered Career Guidance
            </span>
          </div>

          <h1 className="animate-fade-up-delay-1 text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-display font-bold leading-[1.05] tracking-tight mb-6 max-w-4xl">
            <span className="text-foreground">Discover Your</span>
            <br />
            <span className="bg-gradient-to-r from-primary via-glow to-accent bg-clip-text text-transparent">
              Dream Career
            </span>
          </h1>

          <p className="animate-fade-up-delay-2 text-lg sm:text-xl text-muted-foreground max-w-2xl mb-10 leading-relaxed">
            Answer a few questions about your interests, skills, and academics.
            Get AI-powered career recommendations backed by real industry data.
          </p>

          <div className="animate-fade-up-delay-3 flex flex-wrap items-center justify-center gap-4">
            <button
              onClick={onStart}
              className="group relative px-8 py-4 bg-gradient-to-r from-primary to-glow text-primary-foreground font-display font-semibold text-lg rounded-2xl overflow-hidden transition-all duration-300 hover:scale-105 hover:shadow-[0_0_50px_hsl(var(--glow)/0.6)]"
            >
              <span className="relative z-10 flex items-center gap-2">
                Start Free Assessment
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" />
              </span>
            </button>
            <button
              onClick={scrollToHowItWorks}
              className="px-6 py-4 rounded-2xl border border-primary/40 text-foreground hover:border-primary hover:bg-primary/10 font-medium transition-all duration-300 flex items-center gap-2 hover:shadow-[0_0_25px_hsl(var(--primary)/0.2)]"
            >
              See How It Works
              <ChevronDown className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10">
          <div className="flex flex-col items-center gap-2 animate-bounce-slow">
            <span className="text-xs text-muted-foreground/60 uppercase tracking-widest">
              Scroll
            </span>
            <ChevronDown className="w-5 h-5 text-muted-foreground/60" />
          </div>
        </div>

        {/* Bottom fade */}
        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
      </section>

      {/* ════════════════════════ STATS ════════════════════════ */}
      <section className="relative py-20 px-6 overflow-hidden">
        {/* Parallax gradient background */}
        <div className="absolute inset-0 bg-gradient-to-r from-primary/8 via-background to-accent/8" />
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute w-[600px] h-[600px] rounded-full bg-primary/5 blur-[200px] -top-60 left-1/4 animate-pulse-slow" />
          <div className="absolute w-[400px] h-[400px] rounded-full bg-accent/5 blur-[160px] -bottom-40 right-1/4 animate-pulse-slow" style={{ animationDelay: '2s' }} />
        </div>
        {/* Top & bottom dividers */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent" />
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent" />

        <div className="relative z-10 max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-10 md:gap-6">
          {stats.map((stat, i) => (
            <ScrollReveal key={stat.label} delay={i * 0.12} className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <stat.icon className="w-7 h-7 text-primary" />
              </div>
              <div className="text-5xl sm:text-6xl lg:text-7xl font-display font-bold text-foreground mb-2 tracking-tight">
                <CountUp
                  target={stat.value}
                  suffix={stat.suffix}
                  decimals={stat.decimals}
                />
              </div>
              <div className="text-base sm:text-lg text-muted-foreground font-medium">{stat.label}</div>
            </ScrollReveal>
          ))}
        </div>
      </section>

      {/* ════════════════════════ CAREER FIELDS ════════════════════════ */}
      <section className="py-20 px-6 bg-background relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute w-[500px] h-[500px] rounded-full bg-primary/[0.03] blur-[180px] top-0 -right-60" />
        </div>
        <div className="relative z-10 max-w-6xl mx-auto">
          <ScrollReveal className="text-center mb-14">
            <span className="chip mb-4 inline-block">Explore Fields</span>
            <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4">
              8 Career Fields, Infinite Possibilities
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              From STEM to creative arts — our AI matches your unique strengths to the
              field where you&apos;ll thrive.
            </p>
          </ScrollReveal>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {careerFields.map((field, i) => (
              <ScrollReveal key={field.id} delay={i * 0.08}>
                <div className="group glass-card p-6 h-full hover:border-primary/60 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_0_30px_hsl(var(--primary)/0.2)] cursor-default">
                  <div className="w-12 h-12 mb-4 rounded-xl bg-gradient-to-br from-primary/15 to-accent/15 flex items-center justify-center group-hover:scale-110 group-hover:shadow-[0_0_20px_hsl(var(--primary)/0.3)] transition-all duration-300">
                    <field.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="font-display font-semibold text-foreground mb-1 group-hover:text-primary transition-colors duration-300">
                    {field.title}
                  </h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {field.desc}
                  </p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ════════════════════════ HOW IT WORKS ════════════════════════ */}
      <section id="how-it-works" className="py-20 px-6 relative overflow-hidden">
        {/* Gradient background + decorative orbs — parallax */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-background to-accent/20" />
        <div
          ref={howItWorksParallax.ref}
          className="absolute inset-0 overflow-hidden pointer-events-none will-change-transform"
          style={{ transform: `translateY(${howItWorksParallax.offset}px)` }}
        >
          <div className="absolute w-[500px] h-[500px] rounded-full bg-primary/8 blur-[180px] -top-48 -left-48 animate-pulse-slow" />
          <div className="absolute w-96 h-96 rounded-full bg-accent/8 blur-[150px] -bottom-32 -right-32 animate-pulse-slow" style={{ animationDelay: '3s' }} />
          <div className="absolute w-64 h-64 rounded-full bg-glow/6 blur-[100px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
        </div>
        <div className="relative z-10 max-w-5xl mx-auto">
          <ScrollReveal className="text-center mb-14">
            <span className="chip mb-4 inline-block">How It Works</span>
            <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4">
              Three Steps to Your Future
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Our AI analyzes your unique profile and matches you with career paths
              where you&apos;re most likely to thrive.
            </p>
          </ScrollReveal>

          {/* Steps with connector */}
          <div className="relative">
            {/* Horizontal connector line (desktop only) */}
            <div className="hidden md:block absolute top-16 left-[calc(16.67%+24px)] right-[calc(16.67%+24px)] h-0.5 bg-gradient-to-r from-primary/40 via-glow/40 to-accent/40" />

            <div className="grid gap-8 md:grid-cols-3">
              {steps.map((item, i) => (
                <ScrollReveal key={item.step} delay={i * 0.15}>
                  <div className="group relative glass-card p-8 text-center hover:border-primary/60 transition-all duration-300 hover:shadow-[0_0_35px_hsl(var(--primary)/0.2)] hover:-translate-y-2 h-full flex flex-col items-center">
                    {/* Step badge */}
                    <div className="relative z-10 w-12 h-12 rounded-full bg-gradient-to-r from-primary to-glow text-primary-foreground flex items-center justify-center text-sm font-bold font-display mb-6 group-hover:scale-110 group-hover:shadow-[0_0_20px_hsl(var(--primary)/0.5)] transition-all duration-300">
                      {item.step}
                    </div>
                    {/* Icon */}
                    <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-primary/15 to-accent/15 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                      <item.icon className="w-7 h-7 text-primary" />
                    </div>
                    <h3 className="font-display font-semibold text-lg text-foreground mb-2 group-hover:text-primary transition-colors duration-300">
                      {item.title}
                    </h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {item.desc}
                    </p>
                  </div>
                </ScrollReveal>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ════════════════════════ TESTIMONIALS ════════════════════════ */}
      <section className="py-20 px-6 relative overflow-hidden">
        {/* Subtle gradient for visual flow */}
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/[0.03] to-transparent" />
        <div className="relative z-10 max-w-6xl mx-auto">
          <ScrollReveal className="text-center mb-14">
            <span className="chip-accent mb-4 inline-block">
              <Star className="w-3.5 h-3.5 mr-1 inline" />
              Student Reviews
            </span>
            <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4">
              Trusted by Students
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              See how AI-powered guidance has helped students find clarity in their
              career choices.
            </p>
          </ScrollReveal>

          <div className="grid gap-6 md:grid-cols-3">
            {testimonials.map((t, i) => (
              <ScrollReveal key={t.name} delay={i * 0.12}>
                <div className="group glass-card p-6 h-full flex flex-col hover:border-accent/50 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_0_30px_hsl(var(--accent)/0.15)]">
                  {/* Stars */}
                  <div className="flex gap-0.5 mb-4">
                    {Array.from({ length: t.rating }, (_, j) => (
                      <Star
                        key={j}
                        className="w-4 h-4 fill-accent text-accent"
                      />
                    ))}
                  </div>
                  {/* Quote */}
                  <p className="text-foreground/90 leading-relaxed mb-6 italic flex-1">
                    &ldquo;{t.quote}&rdquo;
                  </p>
                  {/* Author */}
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-glow flex items-center justify-center text-primary-foreground text-sm font-bold font-display shrink-0">
                      {t.initials}
                    </div>
                    <div>
                      <p className="text-sm font-medium text-foreground">
                        {t.name}
                      </p>
                      <p className="text-xs text-muted-foreground">{t.role}</p>
                    </div>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ════════════════════════ CLOSING CTA ════════════════════════ */}
      <section className="py-24 px-6 relative overflow-hidden">
        {/* Background — parallax */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-background to-accent/20" />
        <div
          ref={ctaParallax.ref}
          className="absolute inset-0 overflow-hidden pointer-events-none will-change-transform"
          style={{ transform: `translateY(${ctaParallax.offset}px)` }}
        >
          <div className="absolute w-96 h-96 rounded-full bg-primary/10 blur-[150px] -top-40 -right-40" />
          <div className="absolute w-80 h-80 rounded-full bg-accent/10 blur-[120px] -bottom-20 -left-20" />
        </div>

        <ScrollReveal className="relative z-10 max-w-3xl mx-auto text-center">
          <Sparkles className="w-10 h-10 text-accent mx-auto mb-6" />
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold text-foreground mb-6 leading-tight">
            Ready to Discover
            <br />
            <span className="bg-gradient-to-r from-primary via-glow to-accent bg-clip-text text-transparent">
              Your Perfect Career?
            </span>
          </h2>
          <p className="text-lg text-muted-foreground mb-10 max-w-xl mx-auto">
            Take the free 2-minute assessment and get personalized career
            recommendations backed by AI and real industry data.
          </p>
          <button
            onClick={onStart}
            className="group relative inline-flex items-center gap-2 px-10 py-5 bg-gradient-to-r from-primary to-glow text-primary-foreground font-display font-semibold text-lg rounded-2xl transition-all duration-300 hover:scale-105 hover:shadow-[0_0_60px_hsl(var(--glow)/0.5)]"
          >
            Start Your Journey
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" />
          </button>
        </ScrollReveal>
      </section>
    </>
  );
};

export default HeroSection;
