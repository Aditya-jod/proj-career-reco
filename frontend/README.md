# Frontend Structure for Career Path Recommender

This folder will contain the frontend code. The frontend will be built using React, Vite, and Tailwind CSS to provide an interactive user interface for the Career Path Recommender System. It will communicate with the backend FastAPI service to fetch career predictions, university recommendations, and job role suggestions based on user input.

## Planned Structure

```
frontend/
├── public/                # Static assets
├── src/
│   ├── assets/            # Images, fonts, etc.
│   ├── components/        # Reusable UI components
│   ├── pages/             # Page-level components (Home, Login, Signup, Dashboard)
│   ├── services/          # API calls (Fetch wrappers)
│   ├── hooks/             # Custom React hooks
│   ├── context/           # React Context for global state (auth, user, results)
│   ├── App.jsx            # Main app component
│   ├── main.jsx           # Entry point
│   └── index.css          # Tailwind base styles
├── .env                   # Environment variables (API URL, etc.)
├── tailwind.config.js     # Tailwind CSS config
├── postcss.config.js      # PostCSS config
├── package.json           # Project metadata & scripts
└── vite.config.js         # Vite config
```

> Scaffold with Vite + React + Tailwind CSS for a modern, fast, and maintainable frontend.
