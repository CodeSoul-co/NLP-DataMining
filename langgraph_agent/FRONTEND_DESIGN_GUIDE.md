# Frontend Design Guide for THETA

## Current Status

The current frontend is a **minimal HTML prototype** embedded in the backend (`backend/app/static/index.html`). It demonstrates the basic interaction flow but needs to be replaced with a proper React application.

A React project structure has been created in `/frontend/` but requires `npm install` to work (npm not available on current server).

---

## Design Requirements

### 1. Visual Style: Gemini-like Interface

- **Dark theme** (slate-900 background)
- **Large centered logo** on empty state (THETA with Θ symbol)
- **Chat-first interaction** - main interface is a chat window
- **Sidebar navigation** for secondary features
- **Real-time updates** via WebSocket

### 2. Core Pages

| Page | Purpose | Key Components |
|------|---------|----------------|
| **Chat** | Main interaction | Message list, Input box, Quick actions |
| **Projects/Tasks** | Task management | Task list, Progress bars, Status badges |
| **Data** | Dataset browser | Dataset cards, Column info, Size stats |
| **Results** | Training results | Metrics cards, Topic words grid |
| **Visualizations** | View plots | Image gallery, Interactive pyLDAvis iframe |

### 3. Component Hierarchy

```
App
├── Layout
│   ├── Sidebar
│   │   ├── Logo (THETA + Θ)
│   │   ├── NavItems (Chat, Projects, Data, Results, Viz)
│   │   └── StatusIndicator (GPU status)
│   ├── Header
│   │   └── ConnectionStatus (WebSocket)
│   └── MainContent
│       └── <Page />
├── Pages
│   ├── ChatPage
│   │   ├── WelcomeScreen (when no messages)
│   │   ├── MessageList
│   │   │   └── MessageBubble (user/assistant/system)
│   │   ├── TypingIndicator
│   │   └── ChatInput
│   ├── ProjectsPage
│   │   └── TaskCard (status, progress, metrics)
│   ├── DataPage
│   │   └── DatasetCard
│   ├── ResultsPage
│   │   ├── ResultSelector
│   │   ├── MetricsGrid
│   │   └── TopicWordsGrid
│   └── VisualizationsPage
│       ├── VizSelector
│       └── VizViewer (image/iframe)
└── Hooks
    └── useWebSocket
```

---

## API Integration Points

### Chat Flow

```javascript
// User types message
const handleSubmit = async (message) => {
  // 1. Add user message to UI
  addMessage({ role: 'user', content: message });
  
  // 2. Parse intent (can be done client-side or via backend)
  if (message.toLowerCase().includes('train')) {
    // Extract dataset name
    const dataset = message.match(/train\s+(?:on\s+)?(\w+)/i)?.[1];
    
    // 3. Call API to start task
    const response = await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, mode: 'zero_shot', num_topics: 20 })
    });
    const task = await response.json();
    
    // 4. Subscribe to WebSocket for updates
    ws.send(JSON.stringify({ type: 'subscribe', task_id: task.task_id }));
    
    addMessage({ role: 'assistant', content: `Started training: ${task.task_id}` });
  }
};
```

### WebSocket Integration

```javascript
// hooks/useWebSocket.ts
export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(`ws://${location.host}/api/ws`);
    
    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => {
      setIsConnected(false);
      // Auto-reconnect after 3s
      setTimeout(connect, 3000);
    };
    ws.onmessage = (e) => setLastMessage(JSON.parse(e.data));
    
    wsRef.current = ws;
    return () => ws.close();
  }, []);

  const subscribe = (taskId) => {
    wsRef.current?.send(JSON.stringify({ type: 'subscribe', task_id: taskId }));
  };

  return { isConnected, lastMessage, subscribe };
}
```

### Data Fetching

```javascript
// services/api.ts
const API_BASE = '/api';

export const api = {
  // Health
  health: () => fetch(`${API_BASE}/health`).then(r => r.json()),
  
  // Datasets
  getDatasets: () => fetch(`${API_BASE}/datasets`).then(r => r.json()),
  getDataset: (name) => fetch(`${API_BASE}/datasets/${name}`).then(r => r.json()),
  
  // Results
  getResults: () => fetch(`${API_BASE}/results`).then(r => r.json()),
  getResult: (dataset, mode) => fetch(`${API_BASE}/results/${dataset}/${mode}`).then(r => r.json()),
  getTopicWords: (dataset, mode, topK = 10) => 
    fetch(`${API_BASE}/results/${dataset}/${mode}/topic-words?top_k=${topK}`).then(r => r.json()),
  getVisualizations: (dataset, mode) => 
    fetch(`${API_BASE}/results/${dataset}/${mode}/visualizations`).then(r => r.json()),
  
  // Tasks
  getTasks: () => fetch(`${API_BASE}/tasks`).then(r => r.json()),
  getTask: (taskId) => fetch(`${API_BASE}/tasks/${taskId}`).then(r => r.json()),
  createTask: (config) => fetch(`${API_BASE}/tasks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  }).then(r => r.json()),
  cancelTask: (taskId) => fetch(`${API_BASE}/tasks/${taskId}`, { method: 'DELETE' }).then(r => r.json()),
};
```

---

## State Management

Recommended: **Zustand** (already in package.json)

```javascript
// stores/taskStore.ts
import { create } from 'zustand';

interface TaskStore {
  tasks: Record<string, Task>;
  currentTaskId: string | null;
  addTask: (task: Task) => void;
  updateTask: (taskId: string, updates: Partial<Task>) => void;
  setCurrentTask: (taskId: string) => void;
}

export const useTaskStore = create<TaskStore>((set) => ({
  tasks: {},
  currentTaskId: null,
  addTask: (task) => set((state) => ({ 
    tasks: { ...state.tasks, [task.task_id]: task } 
  })),
  updateTask: (taskId, updates) => set((state) => ({
    tasks: { 
      ...state.tasks, 
      [taskId]: { ...state.tasks[taskId], ...updates } 
    }
  })),
  setCurrentTask: (taskId) => set({ currentTaskId: taskId }),
}));
```

---

## UI Components Needed

### 1. MessageBubble

```tsx
interface MessageBubbleProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

// User: right-aligned, blue background
// Assistant: left-aligned, dark background with border
// System: left-aligned, smaller, muted colors
```

### 2. TaskCard

```tsx
interface TaskCardProps {
  taskId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  currentStep: string;
  progress: number;
  metrics?: { topic_coherence_avg: number; topic_diversity_td: number };
}

// Show progress bar when running
// Show metrics when completed
// Show error message when failed
```

### 3. TopicWordsGrid

```tsx
interface TopicWordsGridProps {
  topicWords: Record<string, string[]>;
  maxTopics?: number;
  wordsPerTopic?: number;
}

// Grid of cards, each showing topic ID and word list
```

### 4. MetricsDisplay

```tsx
interface MetricsDisplayProps {
  metrics: {
    topic_coherence_avg: number;
    topic_diversity_td: number;
    topic_diversity_irbo?: number;
  };
}

// Card grid showing key metrics with labels
```

---

## Styling Guidelines

### Colors (Tailwind)

```javascript
// tailwind.config.js
colors: {
  primary: {
    500: '#0ea5e9',  // Sky blue - main accent
    600: '#0284c7',  // Darker for hover
  },
  dark: {
    700: '#334155',  // Borders
    800: '#1e293b',  // Cards
    900: '#0f172a',  // Background
  }
}
```

### Component Classes

```css
/* Cards */
.card { @apply bg-slate-800 border border-slate-700 rounded-xl p-4; }

/* Buttons */
.btn-primary { @apply px-4 py-2 bg-sky-500 text-white rounded-lg hover:bg-sky-600; }
.btn-secondary { @apply px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 border border-slate-600; }

/* Input */
.input-field { @apply w-full px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-sky-500; }

/* Sidebar item */
.sidebar-item { @apply flex items-center gap-3 px-4 py-3 rounded-lg text-slate-400 hover:bg-slate-700 hover:text-white; }
.sidebar-item.active { @apply bg-slate-700 text-white; }
```

---

## Recommended Tech Stack

| Category | Technology | Reason |
|----------|------------|--------|
| Framework | React 18 | Modern, well-supported |
| Build | Vite | Fast dev server |
| Styling | TailwindCSS | Utility-first, dark theme support |
| State | Zustand | Simple, lightweight |
| Icons | Lucide React | Clean, consistent |
| Animation | Framer Motion | Smooth transitions |
| Charts | Recharts | For metrics visualization |

---

## Development Steps

1. **Setup**
   ```bash
   cd /root/autodl-tmp/langgraph_agent/frontend
   npm install
   npm run dev
   ```

2. **Configure proxy** (already in vite.config.ts)
   - `/api` → `http://localhost:8000`
   - `/ws` → `ws://localhost:8000`

3. **Implement pages** in order:
   - ChatPage (core interaction)
   - ProjectsPage (task monitoring)
   - ResultsPage (view outputs)
   - DataPage (browse datasets)
   - VisualizationsPage (view plots)

4. **Add WebSocket integration** for real-time updates

5. **Polish UI** with animations and loading states

---

## Testing Checklist

- [ ] Can list datasets
- [ ] Can start training task
- [ ] WebSocket receives step updates
- [ ] Progress bar updates in real-time
- [ ] Can view completed results
- [ ] Can view topic words
- [ ] Can view/download visualizations
- [ ] Error states handled gracefully
- [ ] Responsive on mobile
