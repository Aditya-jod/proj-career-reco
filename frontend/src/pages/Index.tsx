import { useNavigate, Link } from "react-router-dom";
import { Compass } from "lucide-react";
import HeroSection from "@/components/HeroSection";
import { useAuth } from "@/context/AuthContext";

const Index = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const handleStart = () => {
    if (!isAuthenticated) {
      navigate("/login", { state: { from: "/assessment" } });
    } else {
      navigate("/assessment");
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <HeroSection onStart={handleStart} />
      <footer className="py-10 px-6 bg-gradient-to-b from-primary/5 via-muted/20 to-muted/30 border-t border-border/50">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Compass className="w-5 h-5 text-primary" />
            <span className="font-display font-semibold text-foreground">CareerPath</span>
          </div>
          <p className="text-sm text-muted-foreground text-center">
            &copy; {new Date().getFullYear()} Career Path Recommender
          </p>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <Link to="/login" className="hover:text-primary transition-colors">Log In</Link>
            <Link to="/signup" className="hover:text-primary transition-colors">Sign Up</Link>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;

