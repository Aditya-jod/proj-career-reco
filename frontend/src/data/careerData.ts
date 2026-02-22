/**
 * Career Path Recommendation Data & API Integration
 * Connects frontend forms to backend ML models via FastAPI
 */

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ==================== Type Definitions ====================

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
}

// ==================== Dropdown Options ====================

export const interestOptions = [
  "Technology",
  "Healthcare",
  "Business",
  "Creative Arts",
  "Education",
  "Finance",
  "Engineering",
  "Research",
  "Social Work",
  "Law",
];

export const skillOptions = [
  "Problem Solving",
  "Communication",
  "Leadership",
  "Analytical Thinking",
  "Creativity",
  "Technical Skills",
  "Teamwork",
  "Time Management",
  "Critical Thinking",
  "Adaptability",
];

export const hobbyOptions = [
  "Reading",
  "Sports",
  "Gaming",
  "Music",
  "Drawing/Design",
  "Coding",
  "Writing",
  "Photography",
  "Cooking",
  "Traveling",
];

// ==================== API Functions ====================

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
    // Combine interests, skills, and hobbies into a single text description
    const skillsText = [...interests, ...skills, ...hobbies].join(", ");

    // Build the student profile for API
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
      },
      body: JSON.stringify(profile),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const result: RecommendationResult = await response.json();

    // Transform backend response into CareerPath format for frontend
    return transformToCareerPaths(result);
  } catch (error) {
    // Network errors (server unreachable) → fall back to mock data with a warning.
    // Application-level errors (4xx/5xx already thrown above) → re-throw so the
    // caller can show the user a meaningful error message.
    if (error instanceof TypeError) {
      console.warn("Backend unreachable — using mock recommendations.");
      return getMockRecommendations(academics, interests, skills);
    }
    throw error;
  }
}

// Auth API
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
 * Transform API response into CareerPath objects for display
 */
function transformToCareerPaths(result: RecommendationResult): RecommendationResponse {
  const { career, universities, jobs } = result;

  // Create a CareerPath for the primary career
  const primaryCareer: CareerPath = {
    title: career.career_field,
    field: career.career_field,
    description: `Based on your profile, ${career.career_field} is an excellent match with ${(career.confidence * 100).toFixed(0)}% confidence. This field aligns with your academic strengths and skills.`,
    matchScore: Math.round(career.confidence * 100),
    avgSalary: "$60,000 - $150,000 (varies by role)",
    growth: "Growing market demand",
    skills: extractTopSkills([
      ...universities.universities.slice(0, 3).map((u) => u.name),
      ...jobs.jobs.slice(0, 3).map((j) => j.title),
    ]),
    pathway: [
      {
        stage: "Foundation Phase (Years 0-2)",
        detail: `Begin your ${career.career_field} journey with foundational learning and entry-level positions.`,
      },
      {
        stage: "Growth Phase (Years 2-5)",
        detail: "Develop expertise and take on more responsibility in your chosen field.",
      },
      {
        stage: "Mastery Phase (Years 5+)",
        detail: "Become a leader and mentor in your field with advanced opportunities.",
      },
    ],
  };

  // Create CareerPath entries for alternative careers
  const alternativesCareers: CareerPath[] = career.alternatives.map(
    ([field, confidence], index) => ({
      title: field,
      field: field,
      description: `A strong alternative career path with ${(confidence * 100).toFixed(0)}% match score.`,
      matchScore: Math.round(confidence * 100),
      avgSalary: "$50,000 - $120,000",
      growth: "Stable to growing",
      skills: extractTopSkills([
        `Study ${field}`,
        `Learn ${field} fundamentals`,
        `Build ${field} portfolio`,
      ]),
      pathway: [
        {
          stage: "Explore",
          detail: `Learn more about ${field}`,
        },
        {
          stage: "Learn",
          detail: `Acquire key skills in ${field}`,
        },
        {
          stage: "Execute",
          detail: `Build a career in ${field}`,
        },
      ],
    })
  );

  return {
    careers: [primaryCareer, ...alternativesCareers],
    universities: universities.universities,
    jobs: jobs.jobs,
  };
}

/**
 * Extract top skills from a list of items
 */
function extractTopSkills(items: string[]): string[] {
  const keywords = [
    "Analysis",
    "Design",
    "Leadership",
    "Communication",
    "Technical",
    "Creative",
    "Problem Solving",
    "Management",
  ];

  // Return keywords that appear or a subset of items
  return keywords.filter((k) =>
    items.some((item) => item.toLowerCase().includes(k.toLowerCase()))
  );
}

/**
 * Fallback: Get mock recommendations if API is unavailable
 * This ensures the frontend still works during development
 */
export function getMockRecommendations(
  academics: string,
  interests: string[],
  skills: string[]
): RecommendationResponse {
  const careers: CareerPath[] = [
    {
      title: "Software Engineer",
      field: "Technology",
      description:
        "Build innovative software solutions using cutting-edge technologies. This path is ideal if you enjoy problem-solving and technical work.",
      matchScore: 92,
      avgSalary: "$120,000 - $180,000",
      growth: "15% through 2032",
      skills: [
        "Python",
        "JavaScript",
        "System Design",
        "Problem Solving",
        "Teamwork",
      ],
      pathway: [
        {
          stage: "Entry Level (0-2 years)",
          detail: "Junior Developer role, learning codebase and best practices",
        },
        {
          stage: "Mid Level (2-5 years)",
          detail: "Full-stack development, leading small features",
        },
        {
          stage: "Senior Level (5+ years)",
          detail: "Architecture design, mentoring, technical leadership",
        },
      ],
    },
    {
      title: "Data Scientist",
      field: "Data Science",
      description:
        "Extract insights from large datasets using statistics and machine learning. Perfect for analytical minds.",
      matchScore: 85,
      avgSalary: "$100,000 - $160,000",
      growth: "36% through 2032",
      skills: [
        "Python",
        "Statistics",
        "Machine Learning",
        "SQL",
        "Data Visualization",
      ],
      pathway: [
        {
          stage: "Foundation",
          detail: "Learn Python, SQL, and statistical fundamentals. Work with datasets.",
        },
        {
          stage: "Specialization",
          detail: "Master machine learning algorithms and model building. Handle real-world problems.",
        },
        {
          stage: "Leadership",
          detail: "Lead data initiatives, build teams, influence business decisions.",
        },
      ],
    },
    {
      title: "Business Analyst",
      field: "Business",
      description:
        "Bridge the gap between business needs and technology solutions. Great if you love problem-solving and communication.",
      matchScore: 78,
      avgSalary: "$70,000 - $120,000",
      growth: "11% through 2032",
      skills: [
        "Communication",
        "Data Analysis",
        "Problem Solving",
        "SQL",
        "Excel",
      ],
      pathway: [
        {
          stage: "Junior Analyst",
          detail: "Learn business processes, gather requirements, analyze data.",
        },
        {
          stage: "Analyst",
          detail: "Own project analysis, present findings, recommend solutions.",
        },
        {
          stage: "Senior / Lead",
          detail: "Strategic analysis, team leadership, business impact.",
        },
      ],
    },
  ];
  return { careers, universities: [], jobs: [] };
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
