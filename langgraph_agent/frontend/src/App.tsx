import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import ChatPage from './pages/ChatPage'
import ProjectsPage from './pages/ProjectsPage'
import DataPage from './pages/DataPage'
import ResultsPage from './pages/ResultsPage'
import VisualizationsPage from './pages/VisualizationsPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<ChatPage />} />
          <Route path="projects" element={<ProjectsPage />} />
          <Route path="data" element={<DataPage />} />
          <Route path="results" element={<ResultsPage />} />
          <Route path="visualizations" element={<VisualizationsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
