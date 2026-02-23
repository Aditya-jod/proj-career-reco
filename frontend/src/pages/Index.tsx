import { useNavigate } from "react-router-dom";
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
      <footer className="py-8 text-center text-sm text-muted-foreground border-t border-border">
        <p>Career Path Recommender — Helping students find their future.</p>
      </footer>
    </div>
  );
};

export default Index;

