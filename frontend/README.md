# Frontend Structure for Career Path Recommender

This folder will contain the React (Vite) frontend code.

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
