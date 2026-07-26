# ⛏️ CRISPE Prompt Forge

> **Transform raw ideas into battle-ready CRISPE-framework prompts with AI.**

[![Netlify Status](https://api.netlify.com/api/v1/badges/placeholder/deploy-status)](https://app.netlify.com)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3-06B6D4?logo=tailwindcss)
![Gemma 4](https://img.shields.io/badge/Model-Gemma_4_26B-4285F4?logo=google)
![Zustand](https://img.shields.io/badge/State-Zustand-443E38)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- **CRISPE Framework** — Automatically structures your raw ideas into Context, Role, Instruction, Specification, Performance & Example sections
- **Live Typewriter** — Watch your prompt being crafted in real-time with rotating status messages
- **AI-Powered** — Uses Google's **Gemma 4 26B** model via the Gemini API for high-quality prompt generation
- **Prompt History** — All generated prompts are saved locally in your browser (persisted via localStorage)
- **Search & Jump** — Quickly find past prompts by keyword or scroll to them instantly
- **Copy & Download** — Copy to clipboard or download as `.txt` / `.md`
- **Minecraft Pixel Theme** — Immersive dark UI with pixel-perfect Minecraft-inspired design
- **Villager Sound** — A satisfying "Hmm" villager sound plays when your prompt is forged
- **Mobile Friendly** — Responsive layout with slide-out drawer sidebar for mobile devices
- **Dark Theme Only** — Permanent Cave Mode for the best experience
- **Your Own API Key** — Bring your own Gemini API key — no shared quota, no backend costs

---

## 🚀 Live Demo

Visit the live site: [CRISPE Prompt Forge](https://crispe-ai.netlify.app)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | React 18 + TypeScript |
| **Build Tool** | Vite 5 |
| **Styling** | Tailwind CSS 3 |
| **State** | Zustand (persisted to localStorage) |
| **Animation** | Framer Motion |
| **Icons** | Lucide React |
| **AI Model** | Gemma 4 26B (Google Gemini API) |
| **Fonts** | Press Start 2P (pixel) + Inter (body) |
| **Hosting** | Netlify (static deployment) |

---

## 📁 Project Structure

```
CRISPE-AI/
├── src/
│   ├── api/
│   │   └── promptApi.ts          # Gemini API client
│   ├── components/
│   │   ├── ApiKeyGate.tsx        # API key entry screen
│   │   ├── Button.tsx            # Reusable pixel button
│   │   ├── EmptyState.tsx        # Empty history placeholder
│   │   ├── IconButton.tsx        # Icon action button
│   │   ├── PixelPanel.tsx        # Minecraft-styled panel
│   │   ├── PromptCard.tsx        # Prompt result card
│   │   ├── Sidebar.tsx           # Desktop/mobile sidebar
│   │   ├── SkeletonCard.tsx      # Loading skeleton
│   │   ├── TextField.tsx         # Form input
│   │   └── Toast.tsx             # Toast notification
│   ├── features/
│   │   └── promptGenerator/
│   │       ├── PromptGeneratorForm.tsx  # Composer with streaming
│   │       ├── PromptHistory.tsx         # History list
│   │       └── index.ts
│   ├── store/
│   │   └── promptStore.ts        # Zustand store with localStorage
│   ├── App.tsx                   # Main app entry
│   ├── main.tsx                  # React DOM render
│   ├── types.ts                  # TypeScript types
│   └── index.css                 # Tailwind + custom styles
├── dist/                         # Production build output
├── index.html                    # HTML entry point
├── netlify.toml                  # Netlify deployment config
├── tailwind.config.js            # Tailwind theme config
├── vite.config.ts                # Vite configuration
├── tsconfig.json                 # TypeScript config
└── package.json                  # Dependencies
```

---

## 🧪 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/kushal98457-ctrl/CRISPE-AI.git
cd CRISPE-AI
npm install
```

### 2. Run Development Server

```bash
npm run dev
```

Opens at **http://localhost:5173**

### 3. Enter Your API Key

On first launch, you'll be prompted to enter a **Google Gemini API key**.
Get one free at [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Start Forging

Type a raw idea and click the arrow button (or press `Ctrl+Enter`).
Watch as Gemma 4 crafts a perfect CRISPE-framework prompt in real-time!

---

## 🎮 Usage Guide

1. **Enter API Key** — Paste your Gemini API key on the welcome screen
2. **Type Your Idea** — Describe what you want (e.g., "Build a BTC vs ETC price plotter")
3. **Watch It Forge** — Status messages cycle while the AI works its magic
4. **Review Output** — The CRISPE prompt appears with live typewriter effect
5. **Copy or Download** — Copy to clipboard or save as `.txt` / `.md`
6. **Browse History** — Use the sidebar to search and jump between past prompts
7. **Change API Key** — Click "Change API Key" in the sidebar footer

---

## 🎨 Theming

The app features a **Minecraft-inspired pixel theme** with:

- **Cave Mode** — Dark, atmospheric deepslate and stone textures
- **Pixel-Perfect UI** — Custom block shadows, pixel fonts, and grid backgrounds
- **Mob-Inspired Colors** — Emerald (grass), Diamond (rare), Gold, Redstone, Nether
- **Sound Effects** — Villager "Hmm" sound on successful generation

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs via [GitHub Issues](https://github.com/kushal98457-ctrl/CRISPE-AI/issues)
- Submit pull requests for new features
- Suggest UI/UX improvements

---

## 📄 License

This project is open source under the **MIT License**.

---

## 👨‍💻 Created by

**Kushal H** & **Pranav S**

---

<!-- SEO Tags -->
<meta name="description" content="CRISPE Prompt Forge — Transform raw ideas into professional CRISPE-framework prompts using AI-powered Gemma 4. Built with React, TypeScript, Tailwind CSS, and the Google Gemini API." />
<meta name="keywords" content="CRISPE, prompt engineering, AI prompt generator, React, TypeScript, Gemma 4, Gemini API, Minecraft theme, prompt forge, AI tool" />
<meta name="author" content="Kushal H & Pranav S" />
<meta property="og:title" content="CRISPE Prompt Forge" />
<meta property="og:description" content="Forge professional CRISPE-framework prompts from raw ideas using AI." />
<meta property="og:type" content="website" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="CRISPE Prompt Forge" />
<meta name="twitter:description" content="Transform raw ideas into CRISPE-framework prompts with AI." />
