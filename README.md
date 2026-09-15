<div align="center">

# ⛏️ CRISPE Prompt Forge

### Transform raw ideas into battle-ready CRISPE-framework prompts — powered by AI.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-crispe--ai.netlify.app-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://crispe-ai.netlify.app)

![React](https://img.shields.io/badge/React_18-61DAFB?style=flat-square&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript_5-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite_5-646CFF?style=flat-square&logo=vite&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS_3-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)
![Gemma 4](https://img.shields.io/badge/Gemma_4_26B-4285F4?style=flat-square&logo=google&logoColor=white)
![Zustand](https://img.shields.io/badge/Zustand-443E38?style=flat-square&logo=npm&logoColor=white)
![Framer Motion](https://img.shields.io/badge/Framer_Motion-0055FF?style=flat-square&logo=framer&logoColor=white)
![License](https://img.shields.io/github/license/kushal98457-ctrl/CRISPE-AI?style=flat-square&color=green)

<br/>

*An AI-powered prompt engineering tool that structures your raw ideas into the **CRISPE framework** — Context, Role, Instruction, Specification, Performance & Example — using Google's Gemma 4 26B model. Wrapped in a Minecraft-themed pixel UI with typewriter streaming, sound effects, and full prompt history.*

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **CRISPE Framework** | Auto-structures ideas into Context, Role, Instruction, Specification, Performance & Example |
| ⌨️ **Live Typewriter** | Watch prompts being crafted in real-time with animated streaming output |
| 🤖 **AI-Powered** | Google's **Gemma 4 26B** via the Gemini API for high-quality prompt generation |
| 📜 **Prompt History** | All generated prompts saved locally in your browser via `localStorage` |
| 🔍 **Search & Jump** | Instantly find past prompts by keyword or scroll to them |
| 📋 **Copy & Download** | One-click copy to clipboard or download as `.txt` / `.md` |
| 🎮 **Minecraft Pixel Theme** | Immersive dark UI with pixel-perfect Minecraft-inspired design |
| 🔊 **Villager Sound** | A satisfying *"Hmm"* villager sound plays when your prompt is forged |
| 📱 **Mobile Friendly** | Responsive layout with slide-out drawer sidebar for mobile devices |
| 🌑 **Dark Theme Only** | Permanent Cave Mode for the best experience |
| 🔑 **BYOK** | Bring your own Gemini API key — no shared quota, no backend costs |

---

## 🏗️ Architecture

### System Overview

```mermaid
graph TB
    subgraph Client["🖥️ Client — Browser"]
        direction TB
        UI["🎮 React UI<br/>Minecraft Pixel Theme"]
        Store["🗄️ Zustand Store<br/>State Management"]
        LS["💾 localStorage<br/>Persistent Storage"]
    end

    subgraph API_Layer["☁️ External API"]
        Gemini["🤖 Google Gemini API<br/>Gemma 4 26B Model"]
    end

    subgraph Hosting["🌐 Hosting"]
        Netlify["🚀 Netlify<br/>Static CDN"]
    end

    User(("👤 User")) -->|"Enter raw idea"| UI
    UI -->|"Dispatch actions"| Store
    Store -->|"Persist history"| LS
    LS -->|"Hydrate on load"| Store
    UI -->|"API request via Axios"| Gemini
    Gemini -->|"Streamed CRISPE response"| UI
    Netlify -->|"Serve static assets"| Client

    style Client fill:#1a1a2e,stroke:#00d4aa,stroke-width:2px,color:#fff
    style API_Layer fill:#0d1117,stroke:#4285F4,stroke-width:2px,color:#fff
    style Hosting fill:#0d1117,stroke:#00C7B7,stroke-width:2px,color:#fff
    style User fill:#4285F4,stroke:#fff,stroke-width:2px,color:#fff
    style UI fill:#2d1b69,stroke:#a78bfa,stroke-width:1px,color:#fff
    style Store fill:#1e3a5f,stroke:#60a5fa,stroke-width:1px,color:#fff
    style LS fill:#1e3a5f,stroke:#60a5fa,stroke-width:1px,color:#fff
    style Gemini fill:#1a3c34,stroke:#34d399,stroke-width:1px,color:#fff
    style Netlify fill:#1a3c34,stroke:#00C7B7,stroke-width:1px,color:#fff
```

### Data Flow

```mermaid
flowchart LR
    A["📝 Raw Idea"] --> B["🔑 API Key Gate"]
    B --> C["📤 Prompt Generator"]
    C --> D["☁️ Gemini API"]
    D --> E["📥 CRISPE Response"]
    E --> F["⌨️ Typewriter<br/>Streaming"]
    F --> G["📋 Prompt Card"]
    G --> H["🗄️ History<br/>Sidebar"]

    G --> I["📋 Copy"]
    G --> J["💾 Download<br/>.txt / .md"]
    H --> K["🔍 Search<br/>& Jump"]

    style A fill:#4a1942,stroke:#e879f9,stroke-width:2px,color:#fff
    style B fill:#1e3a5f,stroke:#60a5fa,stroke-width:1px,color:#fff
    style C fill:#1e3a5f,stroke:#60a5fa,stroke-width:1px,color:#fff
    style D fill:#1a3c34,stroke:#34d399,stroke-width:2px,color:#fff
    style E fill:#1a3c34,stroke:#34d399,stroke-width:1px,color:#fff
    style F fill:#2d1b69,stroke:#a78bfa,stroke-width:1px,color:#fff
    style G fill:#2d1b69,stroke:#a78bfa,stroke-width:1px,color:#fff
    style H fill:#1e3a5f,stroke:#60a5fa,stroke-width:1px,color:#fff
    style I fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style J fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style K fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
```

### Component Hierarchy

```mermaid
graph TD
    App["🏠 App.tsx"]
    App --> AKG["🔑 ApiKeyGate"]
    App --> PGF["📝 PromptGeneratorForm"]
    App --> SB["📚 Sidebar"]
    App --> Toast["🔔 Toast"]

    PGF --> TF["✏️ TextField"]
    PGF --> BTN["🖲️ Button"]
    PGF --> TW["⌨️ TypewriterText"]
    PGF --> SM["💬 StatusMessage"]

    SB --> PH["📜 PromptHistory"]
    PH --> PC["🃏 PromptCard"]
    PC --> IB["🔘 IconButton"]
    PH --> ES["📭 EmptyState"]
    PC --> PP["🧱 PixelPanel"]
    PGF --> SK["💀 SkeletonCard"]

    subgraph State["🗄️ State Layer"]
        ZS["Zustand Store<br/>promptStore.ts"]
        PA["API Client<br/>promptApi.ts"]
    end

    PGF -.->|"uses"| ZS
    SB -.->|"reads"| ZS
    PGF -.->|"calls"| PA

    style App fill:#2d1b69,stroke:#a78bfa,stroke-width:2px,color:#fff
    style State fill:#0d1117,stroke:#60a5fa,stroke-width:2px,color:#fff
    style ZS fill:#1e3a5f,stroke:#60a5fa,stroke-width:1px,color:#fff
    style PA fill:#1e3a5f,stroke:#60a5fa,stroke-width:1px,color:#fff
    style AKG fill:#1a3c34,stroke:#34d399,stroke-width:1px,color:#fff
    style PGF fill:#1a3c34,stroke:#34d399,stroke-width:1px,color:#fff
    style SB fill:#1a3c34,stroke:#34d399,stroke-width:1px,color:#fff
    style Toast fill:#1a3c34,stroke:#34d399,stroke-width:1px,color:#fff
    style TF fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style BTN fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style TW fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style SM fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style PH fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style PC fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style IB fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style ES fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style PP fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
    style SK fill:#3b2f1e,stroke:#fbbf24,stroke-width:1px,color:#fff
```

---

## 🧩 What is the CRISPE Framework?

CRISPE is a structured prompt engineering methodology designed to produce consistently high-quality AI outputs:

```
┌─────────────────────────────────────────────────────┐
│  C - Context        → Background info & constraints │
│  R - Role           → Who the AI should act as      │
│  I - Instruction    → The specific task to perform   │
│  S - Specification  → Format, tone & requirements   │
│  P - Performance    → Quality & evaluation criteria  │
│  E - Example        → Sample output for reference    │
└─────────────────────────────────────────────────────┘
```

**CRISPE Prompt Forge** takes your rough idea and automatically generates a complete, structured prompt following this framework — ready to use with any LLM.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Framework** | React 18 + TypeScript 5 |
| **Build Tool** | Vite 5 |
| **Styling** | Tailwind CSS 3 |
| **State Management** | Zustand (persisted to `localStorage`) |
| **Animation** | Framer Motion |
| **Icons** | Lucide React |
| **AI Model** | Gemma 4 26B (Google Gemini API) |
| **HTTP Client** | Axios |
| **Fonts** | Press Start 2P (pixel) + Inter (body) |
| **Hosting** | Netlify (static deployment) |

---

## 📁 Project Structure

```
CRISPE-AI/
├── src/
│   ├── api/
│   │   └── promptApi.ts                  # Gemini API client
│   ├── components/
│   │   ├── ApiKeyGate.tsx                # API key entry screen
│   │   ├── Button.tsx                    # Reusable pixel button
│   │   ├── EmptyState.tsx                # Empty history placeholder
│   │   ├── IconButton.tsx                # Icon action button
│   │   ├── PixelPanel.tsx                # Minecraft-styled panel
│   │   ├── PromptCard.tsx                # Prompt result card
│   │   ├── Sidebar.tsx                   # Desktop/mobile sidebar
│   │   ├── SkeletonCard.tsx              # Loading skeleton
│   │   ├── TextField.tsx                 # Form input
│   │   └── Toast.tsx                     # Toast notification
│   ├── features/
│   │   └── promptGenerator/
│   │       ├── PromptGeneratorForm.tsx    # Composer with streaming
│   │       ├── PromptHistory.tsx          # History list
│   │       └── index.ts
│   ├── store/
│   │   └── promptStore.ts                # Zustand store + localStorage
│   ├── App.tsx                           # Main app entry
│   ├── main.tsx                          # React DOM render
│   ├── types.ts                          # TypeScript types
│   └── index.css                         # Tailwind + custom styles
├── dist/                                 # Production build output
├── index.html                            # HTML entry point
├── netlify.toml                          # Netlify deployment config
├── tailwind.config.js                    # Tailwind theme config
├── vite.config.ts                        # Vite configuration
├── tsconfig.json                         # TypeScript config
└── package.json                          # Dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** ≥ 18
- **npm** ≥ 9
- A free **Google Gemini API key** — [Get one here](https://aistudio.google.com/apikey)

### Installation

```bash
# Clone the repository
git clone https://github.com/kushal98457-ctrl/CRISPE-AI.git

# Navigate to the project directory
cd CRISPE-AI

# Install dependencies
npm install

# Start the development server
npm run dev
```

The app will be available at `http://localhost:5173`.

### Build for Production

```bash
npm run build
npm run preview    # Preview the production build locally
```

---

## 🎮 Usage Guide

1. **Enter API Key** — Paste your Gemini API key on the welcome screen
2. **Type Your Idea** — Describe what you want (e.g., *"Build a BTC vs ETH price plotter"*)
3. **Watch It Forge** — Status messages cycle while the AI works its magic (`Ctrl+Enter` to submit)
4. **Review Output** — The CRISPE prompt appears with live typewriter effect
5. **Copy or Download** — Copy to clipboard or save as `.txt` / `.md`
6. **Browse History** — Use the sidebar to search and jump between past prompts
7. **Change API Key** — Click "Change API Key" in the sidebar footer

---

## 🎨 Theming

The app features a **Minecraft-inspired pixel theme** with:

- 🕳️ **Cave Mode** — Dark, atmospheric deepslate and stone textures
- 🧱 **Pixel-Perfect UI** — Custom block shadows, pixel fonts, and grid backgrounds
- 💎 **Mob-Inspired Colors** — Emerald (grass), Diamond (rare), Gold, Redstone, Nether
- 🔊 **Sound Effects** — Villager *"Hmm"* sound on successful generation

---

## 🌐 Deployment

This project is configured for **Netlify** out of the box:

1. Push your code to GitHub
2. Connect the repo to [Netlify](https://netlify.com)
3. Set build command to `npm run build` and publish directory to `dist`
4. Deploy!

Or use the CLI:

```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m "feat: add amazing feature"`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

Feel free to [open an issue](https://github.com/kushal98457-ctrl/CRISPE-AI/issues) for bug reports or feature requests.

---

## 📄 License

This project is open source under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Created by

**Kushal H** & **Pranav S**

---

<div align="center">

**Built with ❤️ and a Minecraft pickaxe**

[⬆ Back to Top](#️-crispe-prompt-forge)

</div>
