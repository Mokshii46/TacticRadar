/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        radar: {
          bg: '#0b0f19',
          card: '#111827',
          cardHover: '#1f2937',
          border: '#1e293b',
          accent: '#3b82f6',
        },
        tactic: {
          fork: '#f59e0b',
          pin: '#10b981',
          skewer: '#06b6d4',
          discovered: '#8b5cf6',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
