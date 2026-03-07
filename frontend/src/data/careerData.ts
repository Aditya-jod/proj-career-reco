/**
 * Career Path Recommendation Data & API Integration
 * Connects frontend forms to backend ML models via FastAPI
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Return an auth header object if a token is stored, otherwise empty.
 */
function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("authToken");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export interface CareerPath {
  title: string;
  field: string;
  description: string;
  matchScore: number;
  avgSalary: string;
  growth: string;
  skills: string[];
  pathway: { stage: string; detail: string }[];
}

export interface RecommendationResponse {
  careers: CareerPath[];
  universities: University[];
  jobs: Job[];
}

export interface University {
  name: string;
  country: string;
  state?: string | null;
  district?: string | null;
  website: string;
  score: number;
}

export interface Job {
  title: string;
  score: number;
}

export interface StudentProfile {
  mathematics_score: number;
  science_score: number;
  language_arts_score: number;
  social_studies_score: number;
  logical_reasoning: number;
  creativity: number;
  communication: number;
  leadership: number;
  social_skills: number;
  skills_text: string;
  preferred_location: string;
}

export interface RecommendationResult {
  career: {
    career_field: string;
    confidence: number;
    alternatives: [string, number][];
  };
  universities: {
    universities: University[];
    total: number;
  };
  jobs: {
    jobs: Job[];
    total: number;
  };
  // Dynamic career metadata from MongoDB (populated by seed script)
  career_metadata?: {
    career_id: string;
    title: string;
    salary_display: string;
    growth_description: string;
    growth_rate: string;
    skills: string[];
    pathway: { stage: string; detail: string }[];
  } | null;
  alternatives_metadata?: {
    career_id: string;
    title: string;
    salary_display: string;
    growth_description: string;
    growth_rate: string;
    skills: string[];
    pathway: { stage: string; detail: string }[];
  }[];
}

// ── Suggestions interface ──────────────────────────────────────────────
export interface SuggestionsData {
  interests: string[];
  skills: string[];
  hobbies: string[];
  academic_streams: string[];
}

/**
 * Hardcoded suggestion lists for the assessment form.
 * These are curated from the project datasets and kept static
 * to avoid extra DB complexity for a college project.
 */
export const SUGGESTIONS: SuggestionsData = {
  academic_streams: [
    "Science (PCM)",
    "Science (PCB)",
    "Commerce",
    "Arts / Humanities",
    "Computer Science",
    "Vocational / Diploma",
  ],
  interests: [
    "Technology",
    "Healthcare",
    "Business",
    "Finance",
    "Education",
    "Creative Arts",
    "Law",
    "Government",
    "Social Work",
    "Engineering",
    "Data Science",
    "Marketing",
    "Design",
    "Music",
    "Sports",
    "Research",
    "Entrepreneurship",
    "Environmental Science",
    "Media & Communication",
    "Architecture",
  ],
  skills: [
    "Problem Solving",
    "Programming",
    "Communication",
    "Leadership",
    "Critical Thinking",
    "Data Analysis",
    "Teamwork",
    "Public Speaking",
    "Writing",
    "Project Management",
    "Research",
    "Creativity",
    "Negotiation",
    "Time Management",
    "Analytical Skills",
    "Mathematics",
    "Design Thinking",
    "Technical Drawing",
    "Financial Literacy",
    "Machine Learning",
  ],
  hobbies: [
    "Reading",
    "Chess",
    "Robotics",
    "Coding",
    "Painting",
    "Blogging",
    "Photography",
    "Sports",
    "Volunteering",
    "Debating",
    "Music",
    "Gaming",
    "Gardening",
    "Cooking",
    "Travelling",
    "Dancing",
    "Writing",
    "DIY Projects",
    "Film Making",
    "Yoga & Meditation",
  ],
};

/**
 * Health check - verify backend is running
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error("Health check failed:", error);
    return false;
  }
}

/**
 * Get comprehensive recommendations (career + universities + jobs)
 * This is the main endpoint used by the assessment form
 */
export async function getRecommendations(
  academics: string,
  interests: string[],
  skills: string[],
  hobbies: string[],
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
  },
  preferredLocation: string
): Promise<RecommendationResponse> {
  try {
    // Combine stream, interests, skills, and hobbies into a single text description.
    // academics (e.g. "Science (PCB)") is the single strongest signal — include it first.
    const skillsText = [academics, ...interests, ...skills, ...hobbies]
      .filter(Boolean)
      .join(", ");

    const profile: StudentProfile = {
      mathematics_score: scores.mathematics,
      science_score: scores.science,
      language_arts_score: scores.language_arts,
      social_studies_score: scores.social_studies,
      logical_reasoning: scores.logical_reasoning,
      creativity: scores.creativity,
      communication: scores.communication,
      leadership: scores.leadership,
      social_skills: scores.social_skills,
      skills_text: skillsText,
      preferred_location: preferredLocation,
    };

    const response = await fetch(`${API_BASE_URL}/api/recommend`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
      },
      body: JSON.stringify(profile),
    });

    if (!response.ok) {
      if (response.status === 401) {
        // Token expired or invalid — clear stale auth state
        localStorage.removeItem("authToken");
        localStorage.removeItem("authUser");
      }
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const result: RecommendationResult = await response.json();
    return transformToCareerPaths(result);
  } catch (error) {
    console.error("Recommendation request failed:", error);
    throw error;
  }
}

export async function loginUser(email: string, password: string): Promise<{ token: string; userId: string; name: string }> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Login failed" }));
    throw new Error(err.detail || "Login failed");
  }
  return response.json();
}

export async function registerUser(name: string, email: string, password: string): Promise<{ token: string; userId: string; name: string }> {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Registration failed" }));
    throw new Error(err.detail || "Registration failed");
  }
  return response.json();
}

/**
 * Transform API response into CareerPath objects for display.
 * All salary, growth, skills, and pathway data comes from the API
 * (sourced from MongoDB) — nothing is hardcoded here.
 */
function transformToCareerPaths(result: RecommendationResult): RecommendationResponse {
  const { career, universities, jobs } = result;
  const meta = result.career_metadata;
  const altMetas = result.alternatives_metadata ?? [];

  const primaryCareer: CareerPath = {
    title: meta?.title ?? career.career_field,
    field: career.career_field,
    description: `Based on your profile, ${meta?.title ?? career.career_field} is an excellent match with ${(career.confidence * 100).toFixed(0)}% confidence. This field aligns with your academic strengths and skills.`,
    matchScore: Math.round(career.confidence * 100),
    avgSalary: meta?.salary_display ?? "See detailed analysis",
    growth: meta?.growth_description ?? "See market data",
    skills: meta?.skills?.length ? meta.skills : [],
    pathway: meta?.pathway?.length
      ? meta.pathway
      : [{ stage: "Getting Started", detail: `Explore opportunities in ${career.career_field}.` }],
  };

  const alternativesCareers: CareerPath[] = career.alternatives.map(
    ([field, confidence], index) => {
      const altMeta = altMetas[index];
      return {
        title: altMeta?.title ?? field,
        field: field,
        description: `A strong alternative career path with ${(confidence * 100).toFixed(0)}% match score.`,
        matchScore: Math.round(confidence * 100),
        avgSalary: altMeta?.salary_display ?? "See detailed analysis",
        growth: altMeta?.growth_description ?? "See market data",
        skills: altMeta?.skills?.length ? altMeta.skills : [],
        pathway: altMeta?.pathway?.length
          ? altMeta.pathway
          : [{ stage: "Explore", detail: `Learn more about ${field}.` }],
      };
    }
  );

  return {
    careers: [primaryCareer, ...alternativesCareers],
    universities: universities.universities,
    jobs: jobs.jobs,
  };
}

/**
 * Fetch dynamic suggestion lists from the backend (interests, skills, hobbies, streams).
 * These are derived from the real datasets — not hardcoded.
 */
export async function fetchSuggestions(): Promise<SuggestionsData> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/suggestions`);
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch suggestions:", error);
    // Return empty defaults so the form still works without suggestions
    return { interests: [], skills: [], hobbies: [], academic_streams: [] };
  }
}

/**
 * Fetch all career metadata from the backend.
 */
export async function fetchCareers(): Promise<CareerPath[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/careers`);
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    const data = await response.json();
    return data.map((c: any) => ({
      title: c.title ?? c.career_id,
      field: c.career_id,
      description: "",
      matchScore: 0,
      avgSalary: c.salary_display ?? "",
      growth: c.growth_description ?? "",
      skills: c.skills ?? [],
      pathway: c.pathway ?? [],
    }));
  } catch (error) {
    console.error("Failed to fetch careers:", error);
    return [];
  }
}

/**
 * Predict career field from profile (single endpoint)
 */
export async function predictCareer(profile: StudentProfile) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/career-predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
      },
      body: JSON.stringify(profile),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Career prediction failed:", error);
    throw error;
  }
}

/**
 * Get university recommendations
 */
export async function getUniversities(
  query: string,
  country?: string,
  topK: number = 10,
  skillsText: string = ""
) {
  try {
    const params = new URLSearchParams({
      query,
      top_k: topK.toString(),
      skills_text: skillsText,
    });

    if (country) {
      params.append("country", country);
    }

    const response = await fetch(
      `${API_BASE_URL}/api/universities?${params}`,
      {
        method: "POST",
        headers: { ...authHeaders() },
      }
    );

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("University recommendation failed:", error);
    throw error;
  }
}

/**
 * Get job recommendations
 */
export async function getJobs(query: string, topK: number = 10) {
  try {
    const params = new URLSearchParams({
      query,
      top_k: topK.toString(),
    });

    const response = await fetch(`${API_BASE_URL}/api/jobs?${params}`, {
      method: "POST",
      headers: { ...authHeaders() },
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Job recommendation failed:", error);
    throw error;
  }
}
